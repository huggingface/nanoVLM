import torch
import torch.nn as nn


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    @classmethod
    def new(cls, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        return cls(max_batch_size, max_seq_length, n_heads, head_dim, dtype)

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]

        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache

        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out
