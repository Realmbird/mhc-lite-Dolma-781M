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


# ===== 781M on Dolma — match the residual run on HF (Realmbird/mhc-781m-residual) =====
# Same specs that produced the residual baseline:
#   batch_size=4, gradient_accumulation_steps=16, max_iters=10000
#   => 65,536 tokens/iter, ~655M tokens total
# On H100-80GB this clocked ~25.9K tokens/sec ≈ 2.5s/iter ≈ ~7 hours per model.
#
# Prereq (run once, multi-hour download + tokenize):
#   uv run python data/Dolma/prepare.py
#
# mHC-lite:
torchrun --standalone --nproc_per_node=1 train.py \
    config/train_dolma.py config/large_781m.py config/with_mhc_lite.py \
    --batch_size=4 --gradient_accumulation_steps=16 --max_iters=10000 \
    --wandb_run_name=781m-mhc-lite
# mHC:
torchrun --standalone --nproc_per_node=1 train.py \
    config/train_dolma.py config/large_781m.py config/with_mhc.py \
    --batch_size=4 --gradient_accumulation_steps=16 --max_iters=10000 \
    --wandb_run_name=781m-mhc





