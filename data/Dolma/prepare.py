"""
Download a configurable subset of Dolma v1_7 shards from olmo-data.org and
tokenize to train.bin / val.bin (uint16, GPT-2 BPE).

The official `load_dataset("allenai/dolma", "v1_7")` no longer works because
modern `datasets` rejects script-based dataset repos; the HF repo only ships
URL lists. We pull shards directly via HTTPS from the URL list.

Defaults target a 1×A100 single-GPU run of the 781M model: ~2 shards, ~1.5 B
tokens, mixed across two web-heavy sources to roughly approximate Dolma's
actual distribution.
"""
import os
import gzip
import json
import argparse
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import tiktoken
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from multiprocessing import Pool


HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw")
TRAIN_BIN = os.path.join(HERE, "train.bin")
VAL_BIN = os.path.join(HERE, "val.bin")
URL_LIST_REPO = "allenai/dolma"
URL_LIST_FILE = "urls/v1_7.txt"
EOT = 50256  # tiktoken gpt2 eot
DTYPE = np.uint16
VAL_FRAC = 0.001  # ~0.1% held out


def fetch_url_list():
    p = hf_hub_download(repo_id=URL_LIST_REPO, filename=URL_LIST_FILE, repo_type="dataset")
    with open(p) as f:
        return [u.strip() for u in f if u.strip()]


def source_of(url: str) -> str:
    return url.split("/dolma-v1_7/")[1].split("/")[0]


def pick_shards(urls, n_shards: int, sources: Optional[list], seed: int):
    by_src = defaultdict(list)
    for u in urls:
        by_src[source_of(u)].append(u)
    rng = random.Random(seed)
    if sources is None:
        # default: heavy web mix that approximates Dolma's actual mass
        sources = ["falcon-refinedweb-filtered", "cc_en_middle", "cc_en_head", "c4-filtered", "wikiref_megawika"]
    picked = []
    si = 0
    while len(picked) < n_shards:
        src = sources[si % len(sources)]
        pool = by_src.get(src, [])
        if pool:
            u = rng.choice(pool)
            if u not in picked:
                picked.append(u)
        si += 1
        if si > 10_000:
            break
    return picked[:n_shards]


def download(url: str, dest: str):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=os.path.basename(dest), leave=False
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
    os.rename(tmp, dest)
    return dest


_ENC = None


def _init_worker():
    global _ENC
    _ENC = tiktoken.get_encoding("gpt2")


def _encode_batch(texts):
    out = []
    for t in texts:
        ids = _ENC.encode_ordinary(t)
        ids.append(EOT)
        out.append(np.array(ids, dtype=DTYPE))
    return out


def stream_texts(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            txt = obj.get("text")
            if txt:
                yield txt


def tokenize_shard_to_writers(path: str, train_f, val_f, pool, batch: int, rng: random.Random):
    train_tokens = 0
    val_tokens = 0
    buf = []
    pending = []  # list of (split, async_result)

    def drain(force_all=False):
        nonlocal train_tokens, val_tokens
        keep = []
        for split, ar in pending:
            if force_all or ar.ready():
                arrays = ar.get()
                for arr in arrays:
                    if split == "train":
                        train_f.write(arr.tobytes())
                        train_tokens += arr.size
                    else:
                        val_f.write(arr.tobytes())
                        val_tokens += arr.size
            else:
                keep.append((split, ar))
        pending[:] = keep

    for txt in stream_texts(path):
        buf.append(txt)
        if len(buf) >= batch:
            split = "val" if rng.random() < VAL_FRAC else "train"
            ar = pool.apply_async(_encode_batch, (buf,))
            pending.append((split, ar))
            buf = []
            while len(pending) >= 8:
                drain(force_all=False)
                if len(pending) >= 8:
                    pending[0][1].wait()
    if buf:
        split = "val" if rng.random() < VAL_FRAC else "train"
        ar = pool.apply_async(_encode_batch, (buf,))
        pending.append((split, ar))
    drain(force_all=True)
    return train_tokens, val_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_shards", type=int, default=2,
                    help="number of v1_7 shards to download/tokenize (default: 2 ≈ ~1.5 B tokens)")
    ap.add_argument("--sources", type=str, default=None,
                    help="comma-separated source names to round-robin from (default: web-heavy mix)")
    ap.add_argument("--seed", type=int, default=2357)
    ap.add_argument("--num_proc", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=1024,
                    help="texts per tokenization task")
    ap.add_argument("--list_sources", action="store_true",
                    help="print Dolma v1_7 source breakdown and exit")
    args = ap.parse_args()

    urls = fetch_url_list()
    if args.list_sources:
        from collections import Counter
        c = Counter(source_of(u) for u in urls)
        for src, n in c.most_common():
            print(f"{src:30s} {n}")
        print(f"TOTAL: {len(urls)}")
        return

    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None
    picked = pick_shards(urls, args.n_shards, sources, args.seed)
    print(f"selected {len(picked)} shard(s):")
    for u in picked:
        print(f"  {u}")

    os.makedirs(RAW_DIR, exist_ok=True)
    paths = []
    for u in picked:
        name = u.split("/")[-1]
        src = source_of(u)
        local_dir = os.path.join(RAW_DIR, src)
        os.makedirs(local_dir, exist_ok=True)
        dest = os.path.join(local_dir, name)
        print(f"downloading {u} -> {dest}")
        download(u, dest)
        paths.append(dest)

    if os.path.exists(TRAIN_BIN):
        os.remove(TRAIN_BIN)
    if os.path.exists(VAL_BIN):
        os.remove(VAL_BIN)

    rng = random.Random(args.seed)
    total_train = 0
    total_val = 0
    with open(TRAIN_BIN, "ab") as tf, open(VAL_BIN, "ab") as vf, \
         Pool(args.num_proc, initializer=_init_worker) as pool:
        for shard_path in paths:
            print(f"tokenizing {shard_path}")
            t, v = tokenize_shard_to_writers(shard_path, tf, vf, pool, args.batch_size, rng)
            total_train += t
            total_val += v
            print(f"  shard contributed: train+={t:,} tokens, val+={v:,} tokens")

    print("=" * 60)
    print(f"wrote {TRAIN_BIN}: {total_train:,} tokens ({os.path.getsize(TRAIN_BIN)/1e9:.2f} GB)")
    print(f"wrote {VAL_BIN}:   {total_val:,} tokens ({os.path.getsize(VAL_BIN)/1e9:.3f} GB)")
    print("done.")


if __name__ == "__main__":
    main()
