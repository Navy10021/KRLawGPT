import torch
import numpy as np
import os
import math
import time
from contextlib import nullcontext
from KRLawGPT import KRLawGPT, myGPTConfig


# Using huggingface LLMs or not
using_LLMs = False
if using_LLMs:
    model_type = 'skt/kogpt2-base-v2'

# Set Hyper-parameters for KRLawGPT training
my_vocab_size = 1704
batch_size = 4
block_size = 1024
max_iters = 2000
eval_interval = 200
local_iter_num = 0 
best_val_loss = 1e3
log_interval = 20
eval_iters = 200
iter_num = 0
lr_decay_iters = max_iters # ~= max_iters
min_lr = 6e-5

# Optimizer Hyper-parameters
weight_decay = 1e-1
learning_rate = 6e-4
beta1 = 0.9
beta2 = 0.95
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(">> We use {}".format(device))
dtype = 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

gradient_accumulation_steps = 5
grad_clip = 1.0
decay_lr = True         # whether to decay the learning rate
warmup_iters = 200

# Save trained model
out_dir = "./output/"

# Load train / val dataset
data_dir = "./data"
train_data = np.memmap(os.path.join(data_dir, 'train_law.bin'), dtype = np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val_law.bin'), dtype=np.uint16, mode='r')
print("\n>> Legal text dataset is ready !")


# Build Batch
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


# Estimate Loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            #logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out


# Build learning rate decay scheduler with warmup
def get_lr(iter):
    # linear warmup
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    # iter > LR, return min LR
    if iter > lr_decay_iters:
        return min_lr
    # in between, use cosine-decay
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    cosine_eff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + cosine_eff * (learning_rate - min_lr)


# 1. Get our KRLawGPT models
model_args = dict(
    n_layer = 12,
    n_head = 12,
    n_embd = 768,
    block_size = block_size,
    bias = False,
    vocab_size = None,
    dropout = 0.0,
    )

if using_LLMs:
    override_args = dict(dropout = model_args['dropout'])
    model = KRLawGPT.from_pretrained(model_type, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

else:
    model_args['vocab_size'] = my_vocab_size if my_vocab_size is not None else 51200  
    model_config = myGPTConfig(**model_args)
    model = KRLawGPT(model_config)

# 2. Crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

# 3. Using GPU or not
model.to(device)

# 4. intialize a GradScaler(if using CUDA)
scaler = torch.cuda.amp.GradScaler(enabled = (dtype == 'float16')) 

# 5. Get optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate,(beta1, beta2), device)

# 6. Train loop
X, Y = get_batch('train')
t0 = time.time()

while True:
    # Get learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Evaluate the loss on train & val sets
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"   >> Evaluation  :  step {iter_num}  I  train loss {losses['train']:.4f}  I  val loss {losses['val']:.4f} \n")
        
        # Save the best model
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            if iter_num > 0:
                print("   Save the best-trained KRLawGPT to {}".format(out_dir))
                torch.save(model, os.path.join(out_dir, 'KRLawGPT.pt'))
                torch.save(model.state_dict(), os.path.join(out_dir, 'KRLawGPT_state_dict.pt'))

    # Using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # Clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none = True)

    # Print Loss and time per iteration
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item()
        print(f"   II  iter {iter_num}  I  loss {lossf:.4f}  I  time {dt*1000:.2f}ms  II")
    
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break