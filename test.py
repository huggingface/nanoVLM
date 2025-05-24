from models.utils import check_multiple_choice_with_regex, top_k_top_p_filtering
import torch
import torch.nn.functional as softmax
import torch.nn as nn
from transformers import SiglipVisionConfig
from huggingface_hub import hf_hub_download
import safetensors
from models.vision_transformer import ViT
from models.config import VLMConfig
from models.language_model import RotaryEmbedding

# class RotaryEmbedding(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         self.dim = 576 // 9
#         self.base = 10000
#         self.max_seq_len = 8192

#         inv_freq = 1.0 / (self.base ** (2 * torch.arange(0, self.dim, 2).float() / self.dim))
#         self.register_buffer("inv_freq", inv_freq)
    
#     @torch.no_grad()
#     def forward(self, position_ids):
#         batch_size, seq_len = position_ids.shape

#         flat_position_ids = position_ids.reshape(-1).float()
#         flat_position_ids = flat_position_ids.unsqueeze(1) * self.inv_freq.unsqueeze(0)
def rotate_half(x):
    # Given x = [x0, x1, x2, x3], this will return [-x2, -x3, x0, x1]
    x1, x2 = x.chunk(2, dim=-1) # [bs, seq_len, dim//2], [bs, seq_len, dim//2]
    print(x1)
    print(x2)
    return torch.cat((-x2, x1), dim=-1)

x = torch.tensor([
    [[1.0, 2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0, 8.0]],
     
    [[9.0, 10.0, 11.0, 12.0],
     [13.0, 14.0, 15.0, 16.0]]
])

print("Original x:")
print(x.shape)

rotated = rotate_half(x)
print("\nRotated x:")
print(rotated)