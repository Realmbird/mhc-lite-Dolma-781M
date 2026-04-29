import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Token budgets (override via env vars without editing the file)
TRAIN_TOKEN_BUDGET = int(os.environ.get("DOLMA_TRAIN_TOKENS", 8_000_000_000))
VAL_TOKEN_BUDGET = int(os.environ.get("DOLMA_VAL_TOKENS", 10_000_000))
ENCODE_BATCH = int(os.environ.get("DOLMA_ENCODE_BATCH", 1000))
SHUFFLE_BUFFER = int(os.environ.get("DOLMA_SHUFFLE_BUFFER", 10_000))

enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token

print(
    f"Streaming allenai/dolma v1_7, budget: "
    f"train={TRAIN_TOKEN_BUDGET:,} val={VAL_TOKEN_BUDGET:,}"
)

dataset = load_dataset("allenai/dolma", "v1_7", split="train", streaming=True)
dataset = dataset.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER)

data_dir = os.path.dirname(os.path.abspath(__file__))


def write_split(stream_iter, out_path, budget, desc):
    written = 0
    text_buf = []
    pbar = tqdm(total=budget, desc=desc, unit="tok", smoothing=0.05)
    with open(out_path, "wb") as f:
        def flush():
            nonlocal written
            if not text_buf:
                return False
            ids_batch = enc.encode_ordinary_batch(text_buf)
            text_buf.clear()
            for ids in ids_batch:
                ids.append(eot)
                arr = np.asarray(ids, dtype=np.uint16)
                f.write(arr.tobytes())
                written += len(arr)
                pbar.update(len(arr))
                if written >= budget:
                    return True
            return False

        for example in stream_iter:
            text_buf.append(example["text"])
            if len(text_buf) >= ENCODE_BATCH:
                if flush():
                    break
        else:
            flush()
    pbar.close()
    return written


stream_iter = iter(dataset)
val_written = write_split(
    stream_iter, os.path.join(data_dir, "val.bin"), VAL_TOKEN_BUDGET, "val"
)
print(f"val:   {val_written:,} tokens -> val.bin")
train_written = write_split(
    stream_iter, os.path.join(data_dir, "train.bin"), TRAIN_TOKEN_BUDGET, "train"
)
print(f"train: {train_written:,} tokens -> train.bin")
print(f"Done! Files saved in: {data_dir}")
