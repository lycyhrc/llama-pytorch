import torch
import torch.nn as nn 
import torch.nn.functional as F

import math
from dataclasses import dataclass
from typing import Optional


# 定义了 ModelArgs 数据类，用于存储和管理与llama相关的配置参数
@dataclass
class ModelArgs:
    dim: int = 4096  # 隐藏层的维度大小
    n_layers: int = 32  # 编码器层数N
    n_heads: int = 32  # 查询（queries）的头数
    n_kv_heads: Optional[int] * None  # Number of heads for the k an v
    vocab_size: int = -1  # This will be set when we load the tokenizer 词汇表的大小
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size = 32  # 用于KV缓存的最大批处理大小
    max_seq_len = 2048  # 用于KV缓存的最大序列长度

    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float=10000):
    assert head_dim % 2==0, "Dimension must be divisible by 2"
    theta_numerator = torch.arange(0, head_dim, 2).float()

    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    m = torch.arange(seq_len, device=device)

    freqs = torch.outer(m ,theta).float()

    freqs_complex = torch.polar(torch.ones_like(freqs),freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # 将输入张量的最后一个维度分为两部分，每两个值代表一个复数的实部和虚部，这样，每一对连续值将组成一个复数
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # 调整旋转频率张量freqs_complex的形状以匹配x_complex张量的形状，所以我们需要增加批次维度和头维度
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # 将x_complex张量中的每个复数乘以freqs_complex张量中对应的复数，这导致了复数的旋转，如论文中图1所示
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # 将复数转换回实数
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    # 将输出张量的类型调整为与输入张量相同，并转移到指定的设备
    return x_out.type_as(x).to(device)

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
        h = self.token_embedding(tokens)

        # 获取与位置 [start_pos, start_pos + seq_len] 对应的对 (m, theta)
        freqs_complex = self.freqs_complex[start_pos: start_pos+seq_len]

        # 连续应用在所有编码器层
        for layer in self.layers:
            h = layer(h ,start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
