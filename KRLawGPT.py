import torch
import torch.nn as nn
import torch.nn.functional
import math
from dataclasses import dataclass
import inspect
from transformers import GPT2LMHeadModel

torch.manual_seed(2023)

# Gaussian Error Linear Units
def myGelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# Layer normalization
class LayerNorm(nn.Module):

    def __init__(self, n_dim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))
        self.bias = nn.Parameter(torch.zeros(n_dim)) if bias else None

    def forward(self, input):
        return torch.nn.functional.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# Causal Self-Attention : implementation of a multi-headed causal self attention block inspired by Andrej Karpathy’s NanoGPT repository.
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # query, key, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd,  3 * config.n_embd, bias = config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')   # fast attention by PyTorch 2.0 == True
        #self.flash = False
        if self.flash:
            print("Using flash faster self-attention layer.")

        if not self.flash:
            print("Using slow self-attention layer")  
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) # not update
    
    def forward(self, x):
        B, T, C = x.size()                                              # B, T, C == Batch size, seq length, embedding dimmensionality 
        Q, K, V = self.c_attn(x).split(self.n_embd, dim = 2)
        Q = Q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        K = K.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        V = V.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # flash faster causal self-attention : (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            attn_val = torch.nn.functional.scaled_dot_product_attention(Q, K, V,
                                                                        attn_mask = None,
                                                                        dropout_p = self.dropout,
                                                                        is_causal = True)
        # original causal self-attention
        else:
            attn = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1)))
            attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            attn_score = torch.nn.functional.softmax(attn, dim=-1)
            attn_score = self.attn_dropout(attn_score)
            attn_val = attn_score @ V       # (B, nh, T, hs)
        
        attn_val = attn_val.transpose(1, 2).contiguous().view(B, T, C)      # (B, T, C)
        attn_val = self.c_proj(attn_val)
        attn_val = self.resid_dropout(attn_val)

        return attn_val


# Feed-Forward with GELU
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = myGelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x


# Stack & Residual : [ layer norm - causal selfattention - layer norm - MLP ]
class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))     # with residual
        x = x + self.mlp(self.ln_2(x))      # with residual

        return x
    
    
# Set hyper-parameters
@dataclass
class myGPTConfig:
    block_size: int = 1024           # The maximum context legth for predictions
    vocab_size: int = 51200          # GPT-2 (50257) & KoGPT-2 (51200)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = False


# GPT Language Model
class KRLawGPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                 wte = nn.Embedding(config.vocab_size, config.n_embd),
                 wpe = nn.Embedding(config.block_size, config.n_embd),
                 dropout = nn.Dropout(config.dropout),
                 h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                 ln_f = LayerNorm(config.n_embd, bias = config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)      
        # special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean = 0.0, std = 0.02/math.sqrt(2 * config.n_layer))

        print("\n>> KRLawGPT's parameters : %.2fM" % (self.get_num_params() / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def get_num_params(self, non_embedding = True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets = None):
        device = idx.device
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

        # Forward the myGPT model itself
        tok_emb = self.transformer.wte(idx)       # Token embedding : (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)    # Position embedding : (1, T, n_embd)
        x = self.transformer.dropout(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        else:
            logits = self.lm_head(x[:, [-1], :])    # list[-1] : preserve the T-dim
            loss = None

        return logits, loss

    # Decrease the block size
    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

        for block in self.transformer.h:
            if block.attn.flash:
                print("\nCrop is not available. Keep the original block size(1024)")
                continue
            else:
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]      #### block.attn.bias  ###

    # Optimizer
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print("\n>> Our model trains on fused AdamW: {}".format(use_fused))
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = betas, **extra_args)

        return optimizer       

    # Text Generator from trained KRLawGPT
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)     # (B, T)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)      # (B, 1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    
    # Get pre-trained GPT models : gpt2, gpt3, kogpt2
    @classmethod
    def from_pretrained(cls, model_type, override_args = None):
        assert model_type in {'gpt2', 'gpt2-xl', 'skt/kogpt2-base-v2'}
        override_args = override_args or {}
        print("\n>> Loading weight from {} PLMs".format(model_type))

        config_args = {
            'gpt2' : dict(n_layer=12, n_head=12, n_embd=768),               # 124M params
            'gpt2-xl' : dict(n_layer=48, n_head=25, n_embd=1600),           # 1558M params
            'skt/kogpt2-base-v2' : dict(n_layer=12, n_head=12, n_embd=768)  # 125M params
        }[model_type]
        # KoGPT-based vocab_size, block_size, bias
        config_args['vocab_size'] = 51200 
        config_args['block_size'] = 1024 
        config_args['bias'] = True
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        
        config = myGPTConfig(**config_args)
        model = KRLawGPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]         # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model