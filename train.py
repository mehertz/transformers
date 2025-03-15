import torch
import torch.nn as nn
import math
import tiktoken
import argparse

import torch.utils
import torch.utils.data

from torch.utils.data import DataLoader


class SelfAttention(nn.Module):
    def __init__(self, masked, context_length, emb_dim, n_heads, qkv_bias, drop_rate):
        super(SelfAttention, self).__init__()

        self.n_heads = n_heads
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.masked = masked

        self.q_mat = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.k_mat = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.v_mat = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)

        self.out = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(drop_rate)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x):
        Q = self.q_mat(x)
        K = self.k_mat(x)
        V = self.v_mat(x)

        batches, tokens, dims = x.shape

        Qs = Q.view(batches, tokens, self.n_heads, dims // self.n_heads).transpose(1,2)
        Ks = K.view(batches, tokens, self.n_heads, dims // self.n_heads).transpose(1,2)
        Vs = V.view(batches, tokens, self.n_heads, dims // self.n_heads).transpose(1,2)

        attn_scores = (Qs @ Ks.transpose(2, 3)) / ((dims // self.n_heads) ** 0.5)

        if self.masked:
            attn_scores.masked_fill_(self.mask[:tokens, :tokens], -float('inf'))

        attn_weights = self.dropout(torch.softmax(attn_scores, dim=3)) @ Vs

        context = attn_weights.transpose(1,2).contiguous().view(batches, tokens, self.emb_dim)

        return self.out(context)


class MLP(nn.Module):
    def __init__(self, emb_dim, ff_int_dim_mult):
        super(MLP, self).__init__()

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

        self.ln_1 = nn.LayerNorm(emb_dim)

        self.attention = SelfAttention(
            masked=True,
            context_length=context_length, 
            emb_dim=emb_dim, 
            n_heads=n_heads, 
            drop_rate=drop_rate,
            qkv_bias=qkv_bias
        )

        self.dropout_1 = nn.Dropout(drop_rate)

        self.ln_2 = nn.LayerNorm(emb_dim)

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
        x = self.MLP(x)
        x = self.dropout_2(x)

        x = x + orig

        return x


class TransformerStack(nn.Module):
    def __init__(self, n_layers, **transformer_kwargs):
        super(TransformerStack, self).__init__()

        self.layers = nn.ModuleList([Transformer(**transformer_kwargs) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, context_length, emb_dim, ff_int_dim_mult, n_heads, n_layers, drop_rate, qkv_bias):
        super(GPT, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.positional_embedding = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(drop_rate)
        
        self.transformers = TransformerStack(
            n_layers,
            context_length=context_length, 
            emb_dim=emb_dim, 
            ff_int_dim_mult=ff_int_dim_mult, 
            n_heads=n_heads, 
            drop_rate=drop_rate,
            qkv_bias=qkv_bias
        )

        self.ln = nn.LayerNorm(emb_dim)
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


class CrossEntropyLoss(nn.Module):
    def forward(self, input_logits, target_idxs):
        # input logits: minibatch x vocab
        # target idxs: minibatch
        # padding_masks: minibatch x context_size

        target_probs = nn.functional.log_softmax(input_logits, dim=1)
        target_probs = torch.gather(target_probs, dim=1, index=torch.unsqueeze(target_idxs, 1)).squeeze(1)

        return -torch.mean(target_probs)


class TinyStoriesDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, max_length: int, tokenizer, start_story_idx: int = 0, end_story_idx: int = None, padding_token=1):
        self.max_length = max_length
        self.input_sequences = []
        self.target_sequences = []
        self.paddings_added = torch.tensor([], dtype=torch.int)
        
        # Read and split the file into stories
        with open(path, 'r') as f:
            text = f.read()
        stories = text.split('<|endoftext|>')[start_story_idx:len(stories) if end_story_idx is None else end_story_idx]
        
        _eos_token = torch.tensor(tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'}))

        for story in stories:
            tokens = tokenizer.encode(story.strip())

            for start_idx in range(0, max((len(tokens) - max_length) + 1, 1), max_length):
                input_seq = torch.tensor(tokens[start_idx : start_idx + max_length])
                padding_quantity = max_length - len(input_seq)
                input_seq = torch.cat((input_seq, torch.tensor([padding_token] * padding_quantity, dtype=input_seq.dtype)))
                self.paddings_added = torch.cat((self.paddings_added, torch.tensor([padding_quantity], dtype=input_seq.dtype)))

                self.input_sequences.append(input_seq)

                _target_window_start = start_idx + 1
                _target_window_end = _target_window_start + max_length

                _target_window = torch.tensor(tokens[_target_window_start:_target_window_end])
                _is_end_of_sequence = len(_target_window) < max_length

                _target_window_tokens = _target_window if not _is_end_of_sequence else torch.cat([_target_window, _eos_token])
                _target_window_tokens = torch.cat((_target_window_tokens, torch.tensor([padding_token] * padding_quantity, dtype=input_seq.dtype)))

                self.target_sequences.append(_target_window_tokens)

    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self, idx):
        return self.input_sequences[idx], self.target_sequences[idx], self.paddings_added[idx]


def train_gpt(model, batch_size=10, num_epochs=1, learning_rate=0.0004, weight_decay=0.1):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    loss_fn = CrossEntropyLoss()

    context_length = model.positional_embedding.num_embeddings
    ds = TinyStoriesDataset('/teamspace/studios/this_studio/transformers/data/TinyStoriesV2-GPT4-train.txt', context_length, tiktoken.get_encoding("gpt2"), end_story_idx=150)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True) 

    for epoch in range(num_epochs):
        for input, target, paddings in dl:
            optimizer.zero_grad()
            outputs = model(input)

            padding_tokens_removed_outputs = []
            padding_tokens_removed_target = []
            for i in range(batch_size):
                padding_tokens_removed_outputs.append(outputs[i, :None if paddings[i] == 0 else -paddings[i], :])
                padding_tokens_removed_target.append(target[i, :None if paddings[i] == 0 else -paddings[i]])

            padding_tokens_removed_outputs = torch.cat(padding_tokens_removed_outputs)
            padding_tokens_removed_target = torch.cat(padding_tokens_removed_target)
            loss = loss_fn(padding_tokens_removed_outputs, padding_tokens_removed_target)
            loss.backward()
            optimizer.step()
            
            print(loss)


def inference(gpt: GPT, text: str, max_tokens_out: int):
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(tokenizer.encode(text)).unsqueeze(0)

    gpt.eval()
    with torch.no_grad():
        for _ in range(max_tokens_out):
            logits = gpt(tokens)

            idx = torch.argmax(logits[:, -1, :])
            tokens = torch.cat((tokens, torch.tensor([idx]).unsqueeze(0)), dim=1)

    return tokenizer.decode(tokens[0].tolist())


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
