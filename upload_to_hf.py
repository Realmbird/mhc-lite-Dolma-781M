"""
Convert a nanoGPT-style checkpoint into an HF model repo.

Each variant produces its own repo under the configured user, e.g.
    Realmbird/mhc-781m-residual
    Realmbird/mhc-781m-mhc
    Realmbird/mhc-781m-mhc-lite

The repo bundles:
    model.safetensors     - bf16 state_dict
    config.json           - GPTConfig fields
    train_config.json     - the training-time config dict (hparams)
    metrics.json          - {iter_num, best_val_loss}
    README.md             - architecture summary + load instructions
    model.py              - copy of the nanoGPT model definition
    hyper_conn/           - copy of the HC modules
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo, upload_folder

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to ckpt.pt produced by train.py")
    ap.add_argument("--repo_id", required=True, help="HF repo id, e.g. Realmbird/mhc-781m-residual")
    ap.add_argument("--variant", required=True, choices=["residual", "mhc", "mhc-lite"])
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--staging_dir", default=None)
    return ap.parse_args()


def _load_ckpt(ckpt_path):
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def _strip_compile_prefix(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            out[k[len("_orig_mod."):]] = v
        else:
            out[k] = v
    return out


def _to_bf16(sd):
    out = {}
    for k, v in sd.items():
        if v.is_floating_point():
            out[k] = v.to(torch.bfloat16).contiguous()
        else:
            out[k] = v.contiguous()
    return out


def _readme(variant, model_args, train_config, metrics):
    n_params = "≈ 781M (GPT-2 Large shape, 36×1280, 4 streams when HC enabled)"
    method = {
        "residual": "vanilla residual (no hyper-connections)",
        "mhc": "Manifold-Constrained Hyper-Connections (paper-faithful, Sinkhorn-Knopp)",
        "mhc-lite": "mHC-lite (lightweight HC variant from the mhc-lite paper)",
    }[variant]
    val_loss = metrics.get("best_val_loss")
    iter_num = metrics.get("iter_num")
    val_str = f"{float(val_loss):.4f}" if val_loss is not None else "n/a"
    iter_str = str(iter_num) if iter_num is not None else "n/a"

    return f"""---
license: apache-2.0
tags:
- gpt2
- nanogpt
- mhc
- hyper-connections
- dolma
datasets:
- allenai/dolma
---

# mhc-781m-{variant}

A 781M-parameter GPT-2-shape language model trained on a subset of Dolma v1_7.
This is the **{variant}** variant of a 3-way comparison (residual / mhc / mhc-lite).

- Training method: **{method}**
- Architecture: {n_params}
- Tokenizer: GPT-2 BPE (`tiktoken` `gpt2`, vocab 50304)
- Sequence length: 1024
- Optimizer: AdamW (paper used AdamW + Muon; here AdamW only)
- Best validation loss: **{val_str}** at iter {iter_str}

## Training recipe

Recipe follows "Ablate and Rescue" (arxiv 2603.14833) with one deviation:
**effective batch size is reduced** from the paper's 0.5M tokens/step to fit
on a single GPU.

| | this run | paper |
|---|---|---|
| Tokens / step | 65,536 | 524,288 |
| Steps | 10,000 | 10,000+ |
| Tokens seen | ~655M | ~3.18B |
| LR / min_lr | 3e-4 / 3e-5 | 3e-4 / 3e-5 |
| Warmup / decay | 200 / 10K (cosine) | 200 / 10K (cosine) |
| Weight decay | 0.1 | 0.1 |
| β1, β2 | 0.9, 0.95 | 0.9, 0.95 |
| Grad clip | 1.0 | 1.0 |
| Bias | True | True |

## Load and run

This is **not** a `transformers`-native model. The state_dict targets the
nanoGPT-style `GPT(GPTConfig)` class in this repo. To use it:

```python
from huggingface_hub import snapshot_download
import sys, json, torch
from safetensors.torch import load_file

local = snapshot_download(repo_id="{{REPO_ID}}")
sys.path.insert(0, local)
from model import GPT, GPTConfig

with open(f"{{local}}/config.json") as f:
    cfg = GPTConfig(**json.load(f))
model = GPT(cfg)
sd = load_file(f"{{local}}/model.safetensors")
model.load_state_dict(sd)
model.eval()
```

## Companion variants

- [Realmbird/mhc-781m-residual](https://huggingface.co/Realmbird/mhc-781m-residual)
- [Realmbird/mhc-781m-mhc](https://huggingface.co/Realmbird/mhc-781m-mhc)
- [Realmbird/mhc-781m-mhc-lite](https://huggingface.co/Realmbird/mhc-781m-mhc-lite)
"""


def main():
    args = parse_args()
    ckpt = _load_ckpt(args.ckpt)
    sd = _to_bf16(_strip_compile_prefix(ckpt["model"]))
    model_args = ckpt["model_args"]
    train_config = ckpt.get("config", {})
    metrics = {
        "iter_num": ckpt.get("iter_num"),
        "best_val_loss": float(ckpt["best_val_loss"]) if ckpt.get("best_val_loss") is not None else None,
    }

    staging = args.staging_dir or os.path.join(os.path.dirname(args.ckpt), "hf_staging")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging, exist_ok=True)

    save_file(sd, os.path.join(staging, "model.safetensors"))
    with open(os.path.join(staging, "config.json"), "w") as f:
        json.dump(model_args, f, indent=2)
    with open(os.path.join(staging, "train_config.json"), "w") as f:
        json.dump({k: v for k, v in train_config.items()
                   if isinstance(v, (int, float, bool, str, type(None)))}, f, indent=2)
    with open(os.path.join(staging, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    readme = _readme(args.variant, model_args, train_config, metrics)
    readme = readme.replace("{REPO_ID}", args.repo_id)
    with open(os.path.join(staging, "README.md"), "w") as f:
        f.write(readme)

    shutil.copy(os.path.join(REPO_ROOT, "model.py"), os.path.join(staging, "model.py"))
    shutil.copytree(
        os.path.join(REPO_ROOT, "hyper_conn"),
        os.path.join(staging, "hyper_conn"),
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    HfApi()
    create_repo(args.repo_id, private=args.private, exist_ok=True)
    print(f"uploading {staging} -> {args.repo_id}")
    upload_folder(
        repo_id=args.repo_id,
        folder_path=staging,
        commit_message=f"upload {args.variant} 781M checkpoint (best_val_loss={metrics['best_val_loss']})",
    )
    print(f"done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
