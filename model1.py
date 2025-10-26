import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""
    def __init__(self):
        super().__init__()
    
    def forward(self, q, k, v, mask=None):
        dk = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out, attn

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, L_q = q.size(0), q.size(1)
        
        q_proj = self.q_lin(q)
        k_proj = self.k_lin(k)
        v_proj = self.v_lin(v)
        
        q_heads = q_proj.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k_heads = k_proj.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_heads = v_proj.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_out, attn_weights = self.attn(q_heads, k_heads, v_heads, mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        
        return self.out_lin(attn_out)

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    """Transformer Encoder Layer"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        x2 = self.norm1(x)
        attn_out = self.mha(x2, x2, x2, mask=src_mask)
        x = x + self.dropout(attn_out)
        
        x2 = self.norm2(x)
        ff_out = self.ff(x2)
        x = x + self.dropout(ff_out)
        return x

class DecoderLayer(nn.Module):
    """Transformer Decoder Layer"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.encdec_mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        x2 = self.norm1(x)
        self_attn = self.self_mha(x2, x2, x2, mask=tgt_mask)
        x = x + self.dropout(self_attn)
        
        x2 = self.norm2(x)
        encdec_attn = self.encdec_mha(x2, enc_out, enc_out, mask=src_mask)
        x = x + self.dropout(encdec_attn)
        
        x2 = self.norm3(x)
        ff_out = self.ff(x2)
        x = x + self.dropout(ff_out)
        return x

class TransformerSeq2Seq(nn.Module):
    """Complete Transformer Encoder-Decoder"""
    def __init__(self, vocab_size, d_model, num_heads, enc_layers, dec_layers, d_ff, dropout, max_len, pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = pad_id
        
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(enc_layers)
        ])
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(dec_layers)
        ])
        
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        src_emb = self.token_embed(src_ids) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        
        enc_out = src_emb
        for layer in self.enc_layers:
            enc_out = layer(enc_out, src_mask)
        
        tgt_emb = self.token_embed(tgt_ids) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        dec_out = tgt_emb
        for layer in self.dec_layers:
            dec_out = layer(dec_out, enc_out, tgt_mask=tgt_mask, src_mask=src_mask)
        
        logits = self.out_proj(dec_out)
        return logits

    def create_padding_mask(self, seq):
        """Create padding mask: 1 for non-pad tokens, 0 for pad tokens"""
        return (seq != self.pad_id).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, size):
        """Create causal mask for decoder self-attention"""
        mask = torch.tril(torch.ones((size, size)))
        return mask.unsqueeze(0).unsqueeze(1)
