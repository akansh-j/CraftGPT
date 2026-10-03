import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import math

batch_size = 128
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.25
warmup_iters = 100
lr_decay_iters = max_iters
min_lr = 3e-5
grad_clip = 1.0
patience = 3

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def get_lr(iter):
    """Cosine learning rate schedule with linear warmup"""
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    if iter > lr_decay_iters:
        return min_lr
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """Single self-attention head with causal masking"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with projection"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block with pre-norm residual connections"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """GPT-style decoder-only transformer for character-level language modeling"""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Checkpoint system: saves every eval_interval and tracks best model
start_iter = 0
best_val_loss = float('inf')
patience_counter = 0
loss_history = []
checkpoint_path = 'checkpoint.pt'
best_checkpoint_path = 'best_checkpoint.pt'

if torch.cuda.is_available():
    map_location = 'cuda'
else:
    map_location = 'cpu'

try:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['iter'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    patience_counter = checkpoint.get('patience_counter', 0)
    loss_history = checkpoint.get('loss_history', [])
    print(f"✓ Checkpoint loaded from step {checkpoint['iter']}")
    print(f"  Best val loss so far: {best_val_loss:.4f}")
except FileNotFoundError:
    print("No checkpoint found, starting from iteration 0")

print(f"\nTraining on {device}")
print(f"Starting from iteration {start_iter}/{max_iters}")
training_start_time = time.time()

for iter in range(start_iter, max_iters):

    lr = get_lr(iter)
    set_lr(optimizer, lr)

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - training_start_time
        print(f"\nstep {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}, time {elapsed/60:.2f}m")

        loss_history.append({
            'iter': iter,
            'train': losses['train'].item(),
            'val': losses['val'].item()
        })

        checkpoint = {
            'iter': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'loss_history': loss_history,
            'vocab': {'stoi': stoi, 'itos': itos, 'chars': chars},
            'config': {
                'n_embd': n_embd,
                'n_head': n_head,
                'n_layer': n_layer,
                'block_size': block_size,
                'vocab_size': vocab_size,
                'dropout': dropout
            }
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved at step {iter}")

        # Early stopping: track validation loss improvement
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            torch.save(checkpoint, best_checkpoint_path)
            print(f"🌟 Best checkpoint saved at step {iter} (val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} eval(s)")

        if patience_counter >= patience:
            print(f"\n⚠ Early stopping triggered after {patience} evals without improvement")
            print(f"Best val loss: {best_val_loss:.4f}")
            break

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

total_time = time.time() - training_start_time
print(f"\n{'='*60}")
print(f"Training completed in {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Final iteration: {iter}")
print(f"{'='*60}\n")

print("Generating sample text...")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print("\n" + "-"*60)
print(generated)
print("-"*60)

print("\nGenerating full sample to 'more.txt'...")
open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
print("✓ Saved to more.txt")

if loss_history:
    with open('loss_history.txt', 'w') as f:
        f.write("iter,train_loss,val_loss\n")
        for entry in loss_history:
            f.write(f"{entry['iter']},{entry['train']},{entry['val']}\n")
    print("✓ Loss history saved to loss_history.txt")
