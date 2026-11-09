"""
Faz 0 prototipi icin GPT tarzi, decoder-only bir Transformer mimarisi.

Tasarim GTX 1650'nin 4GB VRAM'ine gore kucuk tutuldu: varsayilan config
~4-5M parametre civarinda, egitim sirasinda 1GB'in altinda VRAM kullanir
(4GB'lik kartta rahat bosluk birakir). Boyutlar GPTConfig uzerinden
degistirilebilir; Faz 3'teki asil base pretraining'te ayni kod, daha
buyuk bir config ile (bulut GPU'da) yeniden kullanilacak.

Ogrenilmis (learned) pozisyon embedding'i tercih edildi - RoPE gibi
alternatiflere kiyasla daha az hareketli parca var, hata ayiklamasi
daha kolay. Token embedding ve cikis katmani (lm_head) agirliklari
paylasilir (weight tying) - standart pratik, parametre sayisini azaltir.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4000
    block_size: int = 128   # maksimum context uzunlugu (token)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Gelecekteki pozisyonlari gormeyi engelleyen sabit ust-ucgen maske.
        # Buffer olarak kaydedilir (parametre degil, ama .to(device) ile tasinir).
        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(n_embd, dim=2)

        # (batch, seq_len, n_embd) -> (batch, n_head, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)

        return self.resid_dropout(self.out_proj(out))


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.fc(x))
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-norm + residual: modern GPT mimarilerinin standart deseni,
        # derin ag'larda egitimi daha kararli hale getirir.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape

        assert seq_len <= self.config.block_size, (
            f"Girdi uzunlugu ({seq_len}) block_size'i ({self.config.block_size}) asiyor"
        )

        positions = torch.arange(seq_len, device=input_ids.device)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregresif metin uretimi: her adimda son block_size token'a
        bakip bir sonraki token'i orneklez, uretilen token'i girdiye
        ekleyip tekrarlar.
        """

        self.eval()

        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        self.train()
        return input_ids


if __name__ == "__main__":
    # Hizli bir "yukleniyor mu, VRAM'e sigiyor mu" testi. Egitim dongusu
    # ayri bir dosyada (src/train.py) yazilacak; burada sadece mimarinin
    # kendisi ve GPU uzerinde bir forward+backward adimi dogrulaniyor.

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPTConfig()
    model = GPT(config).to(device)

    print(f"Cihaz: {device}")
    print(f"Parametre sayisi: {model.num_parameters():,}")

    batch_size, seq_len = 32, config.block_size
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    dummy_targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    logits, loss = model(dummy_input, dummy_targets)
    loss.backward()

    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Loss: {loss.item():.4f}")

    if device == "cuda":
        allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Kullanilan VRAM (bu test icin): {allocated_mb:.1f} MB")

    print("Forward + backward basarili.")
