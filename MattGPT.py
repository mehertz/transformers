import torch
import torch.nn as nn
import yaml
import tiktoken
from pydantic import BaseModel, Field
from typing import Optional
from torch.utils.data import DataLoader
import click

import torch.utils
import torch.utils.data


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

        Qs = SelfAttention.rope(Qs, self.rope_cos, self.rope_sin)
        Ks = SelfAttention.rope(Ks, self.rope_cos, self.rope_sin)

        attn_scores = (Qs @ Ks.transpose(2, 3)) / ((dims // self.n_heads) ** 0.5)

        if self.masked:
            attn_scores.masked_fill_(self.mask[:tokens, :tokens], -float('inf'))

        attn_weights = self.dropout(torch.softmax(attn_scores, dim=3)) @ Vs

        context = attn_weights.transpose(1,2).contiguous().view(batches, tokens, self.emb_dim)

        return self.out(context)

    @classmethod
    def rope(cls, x, precomputed_cos, precomputed_sin):
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
    
    @classmethod
    def compute_rope_angles(cls, head_dim, theta_base, context_length):
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

        rope_rot_vals = SelfAttention.compute_rope_angles(transformer_kwargs["emb_dim"] // transformer_kwargs["n_heads"], 10_000, transformer_kwargs["context_length"])

        # Todo: Register the rope_rot_values buffer with this module, and then pass through in the forward implementation.
        # Otherwise, computing here and registering in the MHA implementation means there's no memory saving on model saving
        # and then loading as when loading from disk the rope angles will be stored separately amongst all MHA modules
        self.layers = nn.ModuleList([Transformer(**transformer_kwargs, rope_rot_vals=rope_rot_vals) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

class MattGPT(nn.Module):
    def __init__(self, vocab_size, context_length, emb_dim, ff_int_dim_mult, n_heads, n_layers, drop_rate):
        super(MattGPT, self).__init__()

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

        target_probs = nn.functional.log_softmax(input_logits, dim=1)
        target_probs = torch.gather(target_probs, dim=1, index=torch.unsqueeze(target_idxs, 1)).squeeze(1)

        return -torch.mean(target_probs)


class TinyStoriesDataset(torch.utils.data.IterableDataset):
    def __init__(self, path: str, max_length: int, tokenizer, padding_token=1):
        self.path = path
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.padding_token = padding_token

    def __iter__(self):
        _eos_token = torch.tensor(self.tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'}))

        for story in TinyStoriesDataset.gen_stories(self.path):
            tokens = self.tokenizer.encode(story.strip())

            input_sequences = []
            target_sequences = []
            paddings_added = torch.tensor([], dtype=torch.int)

            for start_idx in range(0, max((len(tokens) - self.max_length) + 1, 1), self.max_length):
                input_seq = torch.tensor(tokens[start_idx : start_idx + self.max_length])
                padding_quantity = self.max_length - len(input_seq)
                input_seq = torch.cat((input_seq, torch.tensor([self.padding_token] * padding_quantity, dtype=input_seq.dtype)))
                paddings_added = torch.cat((paddings_added, torch.tensor([padding_quantity], dtype=input_seq.dtype)))

                input_sequences.append(input_seq)

                _target_window_start = start_idx + 1
                _target_window_end = _target_window_start + self.max_length

                _target_window = torch.tensor(tokens[_target_window_start:_target_window_end])
                _is_end_of_sequence = len(_target_window) < self.max_length

                _target_window_tokens = _target_window if not _is_end_of_sequence else torch.cat([_target_window, _eos_token])
                _target_window_tokens = torch.cat((_target_window_tokens, torch.tensor([self.padding_token] * padding_quantity, dtype=input_seq.dtype)))

                target_sequences.append(_target_window_tokens)

            for i in range (len(input_sequences)):
                yield input_sequences[i], target_sequences[i], paddings_added[i]

    @classmethod
    def gen_stories(cls, path: str):
        buffer = []
    
        with open(path, 'r') as f:
            for line in f:
                if '<|endoftext|>' in line:
                    parts = line.split('<|endoftext|>')
                    buffer.append(parts[0])
                    
                    story = ''.join(buffer)
                    yield story.strip()
                    
                    buffer = [parts[1]] if len(parts) > 1 else []
                else:
                    buffer.append(line)
        
            if buffer:
                final_story = ''.join(buffer)
                if final_story.strip():
                    yield final_story.strip()


def _train(ModelConfig, OptimizerConfig, TrainingConfig):
    model = MattGPT(
        vocab_size=ModelConfig.vocab_size,
        context_length=ModelConfig.context_length,
        emb_dim=ModelConfig.emb_dim,
        ff_int_dim_mult=ModelConfig.ff_int_dim_mult,
        n_heads=ModelConfig.n_heads,
        n_layers=ModelConfig.n_layers,
        drop_rate=ModelConfig.drop_rate
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = OptimizerConfig.learning_rate,
        weight_decay = OptimizerConfig.weight_decay
    )

    dataloader = DataLoader(
        TinyStoriesDataset(
            TrainingConfig.train_filepath,
            ModelConfig.context_length,
            tiktoken.get_encoding("gpt2"),
            padding_token=1
        ),
        batch_size=TrainingConfig.batch_size
    )
    loss_fn = CrossEntropyLoss()

    for epoch in range(TrainingConfig.num_epochs):
        for input, target, paddings in dataloader:
            optimizer.zero_grad()
            outputs = model(input)

            padding_tokens_removed_outputs = []
            padding_tokens_removed_target = []
            for i in range(TrainingConfig.batch_size):
                padding_tokens_removed_outputs.append(outputs[i, :None if paddings[i] == 0 else -paddings[i], :])
                padding_tokens_removed_target.append(target[i, :None if paddings[i] == 0 else -paddings[i]])

            import ipdb; ipdb.set_trace()
            padding_tokens_removed_outputs = torch.cat(padding_tokens_removed_outputs)
            padding_tokens_removed_target = torch.cat(padding_tokens_removed_target)
            loss = loss_fn(padding_tokens_removed_outputs, padding_tokens_removed_target)
            loss.backward()
            optimizer.step()
            
            print(loss)


def _inference(gpt: MattGPT, text: str, max_tokens_out: int):
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(tokenizer.encode(text)).unsqueeze(0)

    gpt.eval()
    with torch.no_grad():
        for _ in range(max_tokens_out):
            logits = gpt(tokens)

            idx = torch.argmax(logits[:, -1, :])
            tokens = torch.cat((tokens, torch.tensor([idx]).unsqueeze(0)), dim=1)

    return tokenizer.decode(tokens[0].tolist())


class ModelConfig(BaseModel):
    vocab_size: int = Field(default=50257, description="Vocabulary size")
    context_length: int = Field(default=1024, description="Context length")
    emb_dim: int = Field(default=768, description="Embedding dimension")
    ff_int_dim_mult: int = Field(default=4, description="Factor increase of embedding dimension for linear layers")
    n_heads: int = Field(default=12, description="Number of attention heads")
    n_layers: int = Field(default=12, description="Number of layers")
    drop_rate: float = Field(default=0.1, description="Dropout rate")


class OptimizerConfig(BaseModel):
    learning_rate: float = Field(default=0.0004, description="Learning rate")
    weight_decay: float = Field(default=0.1, description="Weight decay")


class TrainingConfig(BaseModel):
    train_filepath: str = Field(default="/teamspace/studios/this_studio/transformers/data/TinyStoriesV2-GPT4-train.txt", 
                               description="Path to training data")
    batch_size: int = Field(default=10, description="Batch size")
    num_epochs: int = Field(default=1, description="Number of epochs")


@click.group()
def cli():
    """CLI tool for training and preprocessing data for GPT models."""
    pass


@cli.command()
@click.option("--config", type=click.Path(exists=True), required=True, help="Path to YAML configuration file")
def train(config):
    """Train a GPT model with the configuration specified in a YAML file.
    
    Usage:

    python MattGPT.py train --config config.yaml
    """
    # Load configuration from YAML file
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)
    
    model_cfg = ModelConfig(**cfg['model'])
    optimizer_cfg = OptimizerConfig(**cfg['optimizer'])
    training_cfg = TrainingConfig(**cfg['training'])

    _train(model_cfg, optimizer_cfg, training_cfg)


if __name__ == "__main__":
    cli()
