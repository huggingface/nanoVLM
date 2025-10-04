"""
Test if scaled_dot_product_attention can be exported to ONNX with modern exporter.
Based on GitHub issue: https://github.com/pytorch/pytorch/issues/149662
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSDPAModel(nn.Module):
    """Simple model using scaled_dot_product_attention"""
    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Use scaled_dot_product_attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)

        return out


def test_export():
    print("Creating model...")
    model = SimpleSDPAModel()
    model.eval()

    # Create dummy input
    batch_size = 1
    seq_len = 10
    dim = 64
    dummy_input = torch.randn(batch_size, seq_len, dim)

    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Try exporting with modern dynamo-based export
    print("\nAttempting ONNX export with dynamo=True, opset_version=24...")
    try:
        import onnxscript
        torch.onnx.export(
            model,
            (dummy_input,),
            "test_sdpa.onnx",
            dynamo=True,
            export_params=True,
            opset_version=24
        )
        print("✅ Export succeeded!")

        # Check what opset was actually used
        import onnx
        loaded_model = onnx.load("test_sdpa.onnx")
        actual_opset = loaded_model.opset_import[0].version
        print(f"Actual opset version in exported model: {actual_opset}")

        if actual_opset != 24:
            print(f"⚠️  Warning: Requested opset 24 but got opset {actual_opset}")

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
