"""
Test if scaled_dot_product_attention with both is_causal and attn_mask can be exported to ONNX.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDPAWithMaskModel(nn.Module):
    """Model using scaled_dot_product_attention with both is_causal and attn_mask"""
    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attention_mask):
        """
        Args:
            x: [B, T, C] input embeddings
            attention_mask: [B, T] where 1 = attend, 0 = don't attend (padding)
        """
        B, T, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Create additive attention mask: convert 0/1 mask to -inf/0 mask
        # Shape: [B, T] -> [B, 1, 1, T]
        additive_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * torch.finfo(q.dtype).min

        # Use scaled_dot_product_attention with BOTH is_causal and attn_mask
        # This mimics what nanoVLM does
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=additive_mask,
            dropout_p=0.0,
            is_causal=True
        )

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)

        return out


def test_export():
    print("Creating model with SDPA using both is_causal=True and attn_mask...")
    model = SDPAWithMaskModel()
    model.eval()

    # Create dummy input
    batch_size = 1
    seq_len = 10
    dim = 64
    dummy_input = torch.randn(batch_size, seq_len, dim)
    # Attention mask: all ones (no padding)
    dummy_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Monkey-patch SDPA to combine masks for ONNX compatibility
    import torch.nn.functional as F
    original_sdpa = F.scaled_dot_product_attention

    def onnx_compatible_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        if is_causal and attn_mask is not None:
            batch, n_heads, seq_len_q, _ = query.shape
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_q, device=query.device, dtype=query.dtype) * torch.finfo(query.dtype).min,
                diagonal=1
            ).view(1, 1, seq_len_q, seq_len_q)
            combined_mask = attn_mask + causal_mask
            return original_sdpa(query, key, value, attn_mask=combined_mask, dropout_p=dropout_p, is_causal=False, **kwargs)
        else:
            return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)

    F.scaled_dot_product_attention = onnx_compatible_sdpa

    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input, dummy_mask)
    print(f"Output shape: {output.shape}")

    # Try exporting with modern dynamo-based export
    print("\nAttempting ONNX export with dynamo=True...")
    try:
        torch.onnx.export(
            model,
            (dummy_input, dummy_mask),
            "test_sdpa_with_mask.onnx",
            dynamo=True,
            opset_version=24
        )
        print("✅ Export succeeded!")

        # Check what opset was actually used
        import onnx
        loaded_model = onnx.load("test_sdpa_with_mask.onnx")
        actual_opset = loaded_model.opset_import[0].version
        print(f"Actual opset version: {actual_opset}")

        # Check operators
        print("\nONNX Operators used:")
        op_types = sorted(set([node.op_type for node in loaded_model.graph.node]))
        for op in op_types:
            count = sum(1 for node in loaded_model.graph.node if node.op_type == op)
            print(f"  {op}: {count}")

        return True
    except Exception as e:
        print(f"❌ Export failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_export()
    exit(0 if success else 1)
