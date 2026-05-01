# HANDOFF — 781M mHC on Dolma

**Snapshot date:** 2026-05-01
**Goal:** train 3 variants of GPT-2-Large-shape (781M) on Dolma v1_7, comparing residual / mhc / mhc-lite. Upload each to Hugging Face under `Realmbird/mhc-781m-{variant}`.

## Current state

| Variant | Status | Where |
|---|---|---|
| **residual** | ✅ done @ 10K iters, val 3.63 | https://huggingface.co/Realmbird/mhc-781m-residual |
| **mhc** | 🔄 in-progress, last ckpt @ iter 2000, val 4.31 | `out-dolma-781M-mhc/ckpt.pt` (local) |
| **mhc-lite** | ⏳ not started | — |

## Hardware history

- Trained on: **NVIDIA A100-SXM4 40GB** (single GPU)
- Migrating to: machine with more VRAM (80GB+) — exact GPU TBD

## Files to transfer

| File | Size | Notes |
|---|---|---|
| Repo code | ~50 MB | git push/pull (`Realmbird/mhc-lite-Dolma-781M`) |
| `data/Dolma/train.bin` | 3.7 GB | rsync — 1.94B tokens, falcon + cc_en_middle shards |
| `data/Dolma/val.bin` | 5.4 MB | rsync — 2.8M tokens |
| `out-dolma-781M-mhc/ckpt.pt` | 8.8 GB | rsync — resume from iter 2000 |
| `out-dolma-781M-residual/ckpt.pt` | 8.7 GB | **skip** — already on HF, `snapshot_download` if needed |

**Conversation transcript:** `~/.claude/projects/-home-ubuntu-mhc-lite-Dolma-781M/c39007ed-2032-4a37-8017-a7c122ccafde.jsonl` (back up separately, not in git).

## New-machine setup

```bash
git clone git@github.com:Realmbird/mhc-lite-Dolma-781M.git
cd mhc-lite-Dolma-781M

# uv env
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10 .venv
uv sync

# auth
uv run wandb login              # paste WANDB_API_KEY
uv run huggingface-cli login    # paste HF token (Realmbird account)

# rsync from old machine before launching:
#   rsync -avhP data/Dolma/{train,val}.bin   NEW:~/mhc-lite-Dolma-781M/data/Dolma/
#   rsync -avhP out-dolma-781M-mhc/ckpt.pt   NEW:~/mhc-lite-Dolma-781M/out-dolma-781M-mhc/
```

## Resume mhc training (paper-faithful batch on bigger GPU)

On 80GB GPU, return to the paper's micro-batch (bs=4) and skip the OOM workaround:

```bash
uv run python train.py \
    config/train_dolma.py config/large_781m.py config/with_mhc.py \
    --batch_size=4 --gradient_accumulation_steps=16 \
    --max_iters=10000 --init_from=resume \
    --wandb_run_name=781m-mhc-resumed
```

If 80GB+ available, can go higher (bs=8 accum=8 = 65K tokens/iter same as before, 2× faster).

After it finishes:
```bash
uv run python upload_to_hf.py \
    --ckpt out-dolma-781M-mhc/ckpt.pt \
    --repo_id Realmbird/mhc-781m-mhc \
    --variant mhc
```

Then run mhc-lite the same way:
```bash
uv run python train.py \
    config/train_dolma.py config/large_781m.py config/with_mhc_lite.py \
    --batch_size=4 --gradient_accumulation_steps=16 --max_iters=10000 \
    --wandb_run_name=781m-mhc-lite

uv run python upload_to_hf.py \
    --ckpt out-dolma-781M-mhc-lite/ckpt.pt \
    --repo_id Realmbird/mhc-781m-mhc-lite \
    --variant mhc-lite
```

`chain_resume_hc.sh` does both sequentially with the OOM-workaround settings. Edit `BS=4 ACCUM=16` at the top if you have ≥80GB.

## Key decisions / things that bit us

**1. Dolma v1_7 dataset script broken.** `load_dataset("allenai/dolma", "v1_7")` fails with modern `datasets` (script-based repos rejected). Rewrote `data/Dolma/prepare.py` to download shard URLs directly from `urls/v1_7.txt` and stream-tokenize. Default `--n_shards=2` gives ~1.94 B tokens.

**2. mhc OOM at bs=4 on 40GB.** Smoke test peaked at 34GB but actual training hit 36.85 GB (DDP wrapper from `torchrun --nproc_per_node=1` adds ~2 GB; PyTorch allocator fragmentation adds more). Workaround on 40GB:
- `python train.py` instead of `torchrun` (no DDP wrap)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- bs=2 accum=32 → same 65K tokens/iter, peak ~22.5 GB

On bigger GPU these workarounds are unnecessary — go back to `torchrun` and bs=4+.

**3. Throughput surprise.** Residual sustained 25.9K tok/s (~7 hr for 10K iters). mhc sustained only **2.76K tok/s** — ~9× slower because Sinkhorn-Knopp inner loop runs on every microbatch in every layer. Cold-start microbatch timing is misleading; HC variants don't get the warmup speedup that residual does.

**4. Paper-faithful spec is unreachable on 1× A100-40GB.** Paper uses 0.5M tokens/iter × 10K iters (~5B tokens). Our run uses 65K tokens/iter (8× smaller) → ~655M tokens. This still differentiates the architectures but absolute losses will be higher than paper's. With more VRAM you can move closer to paper batch.

**5. bias=True matters.** Paper Table 2 explicitly says `bias=True`. The repo's other configs default to `bias=False`. `config/large_781m.py` was updated to set `bias=True` explicitly.

## Repo layout (key files)

```
config/
  large_781m.py          # 781M GPT-2-Large shape, bias=True, paper hparams
  train_dolma.py         # Dolma data loader config (eval interval 500, bf16)
  with_mhc.py            # hyper_conn_type="mhc", n=4
  with_mhc_lite.py       # hyper_conn_type="mhc_lite", n=4
data/Dolma/
  prepare.py             # rewritten to download v1_7 shards directly
  train.bin / val.bin    # generated by prepare.py
hyper_conn/              # HC implementations (residual, hc, mhc, mhc_lite, analysis)
  __init__.py            # hyper_conn_init_func() routes by type
  mhc.py / mhc_lite.py   # the two variants
model.py                 # GPT(GPTConfig) — nanoGPT-style with HC wrapper
train.py                 # nanoGPT training loop (supports --init_from=resume)
upload_to_hf.py          # ckpt.pt → HF repo with safetensors + readme + code bundle
chain_train_and_upload.sh   # original sequential train+upload (residual recipe)
chain_resume_hc.sh          # OOM-workaround chain for HC variants
HANDOFF.md               # this file
```

## Auth state on old machine (set up the same on new)

- HF: logged in as `Realmbird` (token in `~/.cache/huggingface/token`)
- WandB: `chriskino777`, project `mhc-lite`, group `781M`
  - residual run: https://wandb.ai/chriskino777-ucla/mhc-lite/runs/7hsoemvw (`781m-residual-6064`)
  - mhc run: https://wandb.ai/chriskino777-ucla/mhc-lite/runs/5ibweuil (`781m-mhc-5467`) — kill before transfer

## What to do first on the new machine

1. `git pull` to get this HANDOFF + all configs/scripts
2. rsync the 3 large files (train.bin, val.bin, mhc ckpt.pt)
3. Set up uv env + wandb + hf auth
4. `nvidia-smi` to confirm GPU type → tell which bs/accum to use
5. Resume mhc with `--init_from=resume`
6. After mhc uploads, run mhc-lite from scratch
