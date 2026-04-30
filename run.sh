# fineweb_edu
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/small_model.py config/with_hc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/small_model.py config/with_mhc_lite.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/small_model.py config/with_mhc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/small_model.py  

torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/medium_model.py config/with_hc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/medium_model.py config/with_mhc_lite.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/medium_model.py config/with_mhc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/medium_model.py  

torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py config/with_hc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py config/with_mhc_lite.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py config/with_mhc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py  


# owt
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py config/with_hc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py config/with_mhc_light.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py config/with_mhc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py  

torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/medium_model.py config/with_hc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/medium_model.py config/with_mhc_lite.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/medium_model.py config/with_mhc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/medium_model.py  

torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py config/with_hc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py config/with_mhc_lite.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py config/with_mhc.py
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py


# ===== 781M on Dolma — single A100-40GB =====
# data prep (run once, multi-hour download + tokenize):
#   uv run python data/Dolma/prepare.py
#
# per-GPU memory budget for 781M (n_layer=36, n_head=20, n_embd=1280) on A100-40GB:
#   batch_size=8, gradient_accumulation_steps=16, block_size=1024 => 131,072 tokens/iter
# (original 8-GPU recipe was 1,048,576 tokens/iter; reduce iter count or accept fewer total tokens)
#
# residual baseline (no HC):
torchrun --standalone --nproc_per_node=1 train.py config/train_dolma.py config/large_781m.py \
    --batch_size=8 --gradient_accumulation_steps=16
# mHC-lite:
torchrun --standalone --nproc_per_node=1 train.py config/train_dolma.py config/large_781m.py config/with_mhc_lite.py \
    --batch_size=8 --gradient_accumulation_steps=16
# mHC:
torchrun --standalone --nproc_per_node=1 train.py config/train_dolma.py config/large_781m.py config/with_mhc.py \
    --batch_size=8 --gradient_accumulation_steps=16





