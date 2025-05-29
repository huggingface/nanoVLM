from typing import Literal, Dict, Type
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class KVCacheBase(ABC, nn.Module):
    """Abstract base class for KV cache implementations."""
    
    def __init__(self, max_batch_size: int, max_seq_length: int, 
                 n_heads: int, head_dim: int, dtype=torch.bfloat16):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype
    
    @abstractmethod
    def update(self, input_pos: torch.Tensor, k_val: torch.Tensor, 
               v_val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the cache with new key-value pairs."""
        pass


class KVCacheFactory:
    """Factory for creating KV cache instances with registry pattern."""
    
    _registry: Dict[str, Type[KVCacheBase]] = {}
    
    @classmethod
    def register(cls, name: str, cache_class: Type[KVCacheBase]) -> None:
        """Register a new KV cache implementation."""
        if not issubclass(cache_class, KVCacheBase):
            raise TypeError(
                f"Cache class {cache_class} must inherit from KVCacheBase"
            )
        cls._registry[name] = cache_class
    
    @classmethod
    def get_available_implementations(cls) -> list[str]:
        """Get list of available KV cache implementations."""
        return list(cls._registry.keys())
    
    @classmethod
    def create_kv_cache(
        cls,
        implementation: str,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype=torch.bfloat16,
    ) -> KVCacheBase:
        """Create a KV cache instance of the specified implementation."""
        if implementation not in cls._registry:
            available = ", ".join(cls.get_available_implementations())
            raise ValueError(
                f"Unknown KV cache implementation '{implementation}'. "
                f"Available implementations: {available}"
            )
        
        cache_class = cls._registry[implementation]
        return cache_class(
            max_batch_size, max_seq_length, n_heads, head_dim, dtype
        )
    
    @staticmethod
    def get_kv_cache(
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype=torch.bfloat16,
        kv_cache_implementation: Literal["static", "dynamic"] = "static",
    ) -> KVCacheBase:
        """Legacy method for backward compatibility."""
        return KVCacheFactory.create_kv_cache(
            kv_cache_implementation,
            max_batch_size,
            max_seq_length,
            n_heads,
            head_dim,
            dtype,
        )


# Reference: https://github.com/pytorch-labs/gpt-fast/blob/main/mixtral-moe/model.py
class StaticKVCache(KVCacheBase):
    """Static KV cache with pre-allocated memory."""
    
    def __init__(self, max_batch_size: int, max_seq_length: int, 
                 n_heads: int, head_dim: int, dtype=torch.bfloat16):
        super().__init__(max_batch_size, max_seq_length, n_heads, head_dim, dtype)
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos: torch.Tensor, k_val: torch.Tensor, 
               v_val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache at specified positions."""
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2], (
            f"Position length {input_pos.shape[0]} != "
            f"key sequence length {k_val.shape[2]}"
        )

        k_out = self.k_cache
        v_out = self.v_cache

        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class DynamicKVCache(KVCacheBase):
    """Dynamic KV cache that grows as needed."""

    def __init__(self, max_batch_size: int, max_seq_length: int, 
                 n_heads: int, head_dim: int, dtype=torch.bfloat16):
        super().__init__(max_batch_size, max_seq_length, n_heads, head_dim, dtype)
        self.k_cache = None
        self.v_cache = None

    def update(self, input_pos: torch.Tensor, k_val: torch.Tensor, 
               v_val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache by concatenating new values."""
        if self.k_cache is None:
            self.k_cache = k_val
            self.v_cache = v_val
            return self.k_cache, self.v_cache

        new_k = torch.cat([self.k_cache, k_val], dim=2)
        new_v = torch.cat([self.v_cache, v_val], dim=2)

        self.k_cache = new_k
        self.v_cache = new_v

        return self.k_cache, self.v_cache


# Register implementations
KVCacheFactory.register("static", StaticKVCache)
KVCacheFactory.register("dynamic", DynamicKVCache)