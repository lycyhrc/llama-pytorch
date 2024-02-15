import torch
import torch.nn as nn 
import torch.nn.functional as F

import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # Number of heads for the queries
    n_kv_heads: Optional[int] * None  # Number of heads for the k an v
    vocab_size: int = -1  # This will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size = 32
    max_seq_len = 2048

    device: str = None


class Transformer(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        
        assert args.vocab_size != -1 ,"vocab size must be set."

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.token_embedding  = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norms = RMSNorm(args.dim, eps = args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias = False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim //self.args.n_heads,
                                                              self.args.max_seq_len*2,
                                                              device = self.args.device)
        
    def forword(self, tokens: torch.Tensor, start_pos: int):
        # (B, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        # (B, seq_len) --> (B, seq_len, Dim)
        H = self.token_embedding(tokens)

        # 获取与位置 [start_pos, start_pos + seq_len] 对应的对 (m, theta)
        freqs_complex = self.freqs_complex[start_pos: start_pos+seq_len]

        # 连续应用在所有编码器层
        for layer in self.layers:
            h = layer(h ,start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
