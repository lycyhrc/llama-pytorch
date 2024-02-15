import torch
import torch.nn as nn 
import torch.nn.functional as F

import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_heads: int = 32  # Number of heads for the queries
    n_kv_heads: Optional[int] * None  # Number of heads for the k an v
    vocab_size: int = -1  # This will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None

    # Needed for KV cache
    max_batch_size = 32
    max_seq_len = 2048

    device: str = None
    