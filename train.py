import torch
import torch.nn as nn
import math
import argparse


class SelfAttention(nn.Module):
    def __init__(self, masked, context_length, emb_dim, n_heads, qkv_bias, drop_rate):
        super(SelfAttention, self).__init__()

        self.n_heads = n_heads
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.masked = masked

        self.q_mat = nn.Linear(emb_dim, emb_dim)
        self.k_mat = nn.Linear(emb_dim, emb_dim)
        self.v_mat = nn.Linear(emb_dim, emb_dim)

        self.out = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(drop_rate)

        self.mask = torch.triu(
            torch.ones(context_length, context_length), diagonal=1
        ).bool()

    def forward(self, x):
        Q = self.q_mat(x)
        K = self.k_mat(x)
        V = self.v_mat(x)

        batches, tokens, dims = x.shape

        Qs = Q.view(batches, tokens, self.n_heads, dims // self.n_heads).transpose(1,2)
        Ks = K.view(batches, tokens, self.n_heads, dims // self.n_heads).transpose(1,2)
        Vs = V.view(batches, tokens, self.n_heads, dims // self.n_heads).transpose(1,2)

        attn_scores = Qs @ Ks.transpose(2, 3) / ((dims // self.n_heads) ** 0.5)

        if self.masked:
            attn_scores.masked_fill_(self.mask, -float('inf'))

        attn_weights = self.dropout(torch.softmax(attn_scores, dim=3)) @ Vs

        context = attn_weights.transpose(1,2).contiguous().view(batches, tokens, self.emb_dim)

        return self.out(context)


class MLP(nn.Module):
    def __init__(self, emb_dim, ff_int_dim_mult):
        self.in_ff = nn.Linear(emb_dim, emb_dim * ff_int_dim_mult)
        self.out_ff = nn.Linear(emb_dim * ff_int_dim_mult, emb_dim)

        self.layers = nn.Sequential(
            self.in_ff,
            nn.GELU(),
            self.out_ff
        )

    def forward(self, x):
        return self.layers(x)



class Transformer(nn.Module):
    def __init__(self, context_length, emb_dim, ff_int_dim_mult, n_heads, drop_rate, qkv_bias):
        super(Transformer, self).__init__()

        self.ln_1 = nn.LayerNorm((context_length, emb_dim))

        self.attention = SelfAttention(
            masked=True,
            context_length=context_length, 
            emb_dim=emb_dim, 
            n_heads=n_heads, 
            drop_rate=drop_rate,
            qkv_bias=qkv_bias
        )

        self.dropout_1 = nn.Dropout(drop_rate)

        self.ln_2 = nn.LayerNorm((context_length, emb_dim))

        self.MLP = MLP(emb_dim, ff_int_dim_mult)

        self.dropout_2 = nn.Dropout(drop_rate)

    def forward(self, x):
        orig = x

        x = self.ln_1(x)
        x = self.attention(x)
        x = self.dropout_1(x)

        x = x + orig

        orig = x

        x = self.ln_2(x)
        x = self.MLP
        x = self.dropout_2(x)

        x = x + orig

        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, context_length, emb_dim, ff_int_dim_mult, n_heads, n_layers, drop_rate, qkv_bias):
        super(GPT, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.positional_embedding = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(drop_rate)

        self.transformers = nn.Sequential(
            [
                Transformer(
                    context_length=context_length, 
                    emb_dim=emb_dim, 
                    ff_int_dim_mult=ff_int_dim_mult, 
                    n_heads=n_heads, 
                    drop_rate=drop_rate,
                    qkv_bias=qkv_bias
                ) 
            ] * n_layers
        )

        self.ln = nn.LayerNorm()
        self.output = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, x):
        batches, context_length = x.shape

        token_embeddings = self.embedding(x)
        positional_embeddings = self.positional_embedding(torch.arange(context_length))

        embeddings = token_embeddings + positional_embeddings

        x = self.dropout(embeddings)
        x = self.transformers(x)
        x = self.ln(x)
        
        return self.output(x)


def train_gpt(config):
    # Define the GPT model
    model = GPT(config)



if __name__ == "__main__":
    sa = SelfAttention(
        masked=True, 
        context_length=4,
        emb_dim=12,
        n_heads=2,
        qkv_bias=False,
        drop_rate=0.1
    )
    sa.forward(torch.rand(1, 4, 12))

    parser = argparse.ArgumentParser(description="Train a GPT model.")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length")
    parser.add_argument("--emb_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--ff_int_dim_mult", type=int, default=4, help="Factor increase of embedding dimension for linear layers")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--drop_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--qkv_bias", type=bool, default=False, help="Query-Key-Value bias")
    args = parser.parse_args()

    # Define the GPT configuration using the parsed arguments
    GPT_CONFIG = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "ff_int_dim_mult": args.ff_int_dim_mult,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": args.qkv_bias
    }

    train_gpt(GPT_CONFIG)
