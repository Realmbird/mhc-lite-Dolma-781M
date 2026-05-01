#!/usr/bin/env bash
# H100-80GB chain: train mhc -> upload -> train mhc-lite -> upload.
# Uses torchrun (DDP wrap is fine with this much VRAM) and the OWT-original
# bs=4 accum=16 = 65,536 tokens/iter (matches what residual was trained on,
# so the 3-way comparison stays apples-to-apples).
set -euo pipefail
cd "$(dirname "$0")"

HF_USER="${HF_USER:-Realmbird}"
BS="${BS:-4}"
ACCUM="${ACCUM:-16}"
ITERS="${ITERS:-10000}"

TRAIN_BIN="data/Dolma/train.bin"
VAL_BIN="data/Dolma/val.bin"
echo "[chain] waiting for $TRAIN_BIN and $VAL_BIN ..."
while [[ ! -s "$TRAIN_BIN" || ! -s "$VAL_BIN" ]]; do
  sleep 30
done
while pgrep -f "data/Dolma/prepare.py" >/dev/null 2>&1; do
  sleep 30
done
echo "[chain] data ready: $(du -h $TRAIN_BIN $VAL_BIN | tr '\n' ' ')"

run_one () {
  local label="$1"           # mhc | mhc-lite
  local extra_cfg="$2"       # config/with_*.py
  local out_dir="out-dolma-781M-${label}"
  local repo_id="${HF_USER}/mhc-781m-${label}"
  local log="${out_dir}/train.log"
  mkdir -p "$out_dir"

  echo "[chain] === ${label} starting at $(date -Iseconds) ==="
  uv run torchrun --standalone --nproc_per_node=1 train.py \
      config/train_dolma.py config/large_781m.py ${extra_cfg} \
      --batch_size=${BS} --gradient_accumulation_steps=${ACCUM} --max_iters=${ITERS} \
      --wandb_run_name="781m-${label}" \
      2>&1 | tee "$log"

  if [[ ! -f "$out_dir/ckpt.pt" ]]; then
    echo "[chain] FATAL: $out_dir/ckpt.pt missing; aborting"
    exit 1
  fi

  echo "[chain] === ${label} uploading to ${repo_id} ==="
  uv run python upload_to_hf.py \
      --ckpt "$out_dir/ckpt.pt" \
      --repo_id "$repo_id" \
      --variant "$label"

  echo "[chain] === ${label} done at $(date -Iseconds) ==="
}

uv run python -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null
uv run python -c "import wandb; wandb.Api().viewer.username" >/dev/null || true

run_one mhc      "config/with_mhc.py"
run_one mhc-lite "config/with_mhc_lite.py"

echo "[chain] HC variants done."
