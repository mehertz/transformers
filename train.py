import torch
import torch.nn as nn
import math
import tiktoken
import argparse

import torch.utils
import torch.utils.data

from torch.utils.data import DataLoader


def compute_rope_angles(head_dim, theta_base, context_length):
    # RoPE is... non-trivial
    # It takes the input key, all the dimensions of the input key, and pairs one dimension up 
    # with another such that you reduce it to head_dim/2 number of 2D pairs.
    # 
    # Each input key is expected to come from an attention head
    # And we split all the dimensions of the key into pairs such that dim 0 is paired with dim N/2 - 
    # effectively (ABCABC)
    #
    # The rotation itself is applied as if we're rotating a complex number
    # To rotate a complex number (z = a + bi), you can multiply it by (cos(angle) + i*sin(angle))
    # So effectively we're doing (a + bi)(cos(angle) + i·sin(angle))
    # a and b are the two "paired" values (AA, BB)
    # The only additional trick is rather than multiplying the pairs by angle, we multiply by 
    # m*angle, where m is the token position.
    # 
    # The key here is that we can pre-compute the cos and sin for the rotation frequencies for each 2D
    # pair up-front. The rotation frequency for each 2D pair is defined as theta_base^(-2i/head_dim) 
    # where i is the index of that 2D pair. We then multiply that dimension-pair angle by the token 
    # index to imbue with positional information which.

    dim_rotation_frequencies = theta_base ** (-2*(torch.arange(0, head_dim / 2) / head_dim))

    positions = torch.arange(context_length)

    # We want to multiply the rotation frequences ([head_dim/2]) by all possible positions
    # And get a [context_length, head_dim] matrix where row0 is dim_rotation_frequencies * 0
    # and row1 is dim_rotation_frequencies * 1, and row2 is....

    positions = positions[:, None]  # [context_length, 1] (i.e. [[0], [1], [2]...])
    dim_rotation_frequencies = dim_rotation_frequencies.unsqueeze(0)  # [1, head_dim/2]
    angles = positions * dim_rotation_frequencies # [context_length, head_dim/2]
    angles = torch.cat([angles, angles], dim=1)  # [context_length, head_dim]

    return torch.cos(angles), torch.sin(angles)


def rope(x, precomputed_cos, precomputed_sin):
    # x [batch, heads, tokens, head_dim]

    _batch_size, _num_heads, seq_len, head_dim = x.shape

    lhs_of_paired_dims = x[:, :, :, :head_dim // 2]
    rhs_of_pairsed_dims = x[:, :, :, head_dim // 2:]

    cos = precomputed_cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = precomputed_sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Right, rotation of a complex number (a + bi) by (angle) = 
    # (a + bi)(cos(angle) + i·sin(angle)) =
    # (a·cos(angle) - b·sin(angle)) + i(a·sin(angle) + b·cos(angle))
    # (note the minus due to i being squared)
    # so for cos, we want to multiply the angle by everything (both parts)
    # but for sin, we want the rhs to be negative

    sine_part = torch.cat((-rhs_of_pairsed_dims, lhs_of_paired_dims), dim=-1) * sin
    cosine_part = cos * x  

    return (cosine_part + sine_part).to(dtype=x.dtype)


class SelfAttention(nn.Module):
    def __init__(self, masked, context_length, emb_dim, n_heads, drop_rate, rope_rot_vals):
        super(SelfAttention, self).__init__()

        self.n_heads = n_heads
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.masked = masked

        self.q_mat = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k_mat = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v_mat = nn.Linear(emb_dim, emb_dim, bias=False)

        self.out = nn.Linear(emb_dim, emb_dim, bias=False)

        self.dropout = nn.Dropout(drop_rate)

        self.register_buffer(
            "rope_cos",
            rope_rot_vals[0]
        )

        self.register_buffer(
            "rope_sin",
            rope_rot_vals[1]
        )

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

        Qs = rope(Qs, self.rope_cos, self.rope_sin)
        Ks = rope(Ks, self.rope_cos, self.rope_sin)

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
    def __init__(self, context_length, emb_dim, ff_int_dim_mult, n_heads, drop_rate, rope_rot_vals):
        super(Transformer, self).__init__()

        self.ln_1 = nn.LayerNorm(emb_dim)

        self.attention = SelfAttention(
            masked=True,
            context_length=context_length, 
            emb_dim=emb_dim, 
            n_heads=n_heads, 
            drop_rate=drop_rate,
            rope_rot_vals=rope_rot_vals
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

        rope_rot_vals = compute_rope_angles(transformer_kwargs["emb_dim"] // transformer_kwargs["n_heads"], 10_000, transformer_kwargs["context_length"])

        # Todo: Register the rope_rot_values buffer with this module, and then pass through in the forward implementation.
        # Otherwise, computing here and registering in the MHA implementation means there's no memory saving on model saving
        # and then loading as when loading from disk the rope angles will be stored separately amongst all MHA modules
        self.layers = nn.ModuleList([Transformer(**transformer_kwargs, rope_rot_vals=rope_rot_vals) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, context_length, emb_dim, ff_int_dim_mult, n_heads, n_layers, drop_rate):
        super(GPT, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.register_buffer("context_length", torch.tensor(context_length, dtype=torch.int64))
        
        self.transformers = TransformerStack(
            n_layers,
            context_length=context_length, 
            emb_dim=emb_dim, 
            ff_int_dim_mult=ff_int_dim_mult, 
            n_heads=n_heads, 
            drop_rate=drop_rate,
        )

        self.ln = nn.LayerNorm(emb_dim)
        self.output = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
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

    context_length = int(model.context_length)
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
    }

    train_gpt(GPT_CONFIG)
