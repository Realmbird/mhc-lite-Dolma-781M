wandb_group = "781M"
out_prefix_model = "781M"

# matches "Ablate and Rescue" paper Table 2: GPT-2 Large shape, bias=True, 4-stream HC
n_layer = 36
n_head = 20
n_embd = 1280
dropout = 0.0
bias = True

learning_rate = 3e-4
min_lr = 3e-5
max_iters = 10000
lr_decay_iters = 10000
warmup_iters = 200
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

