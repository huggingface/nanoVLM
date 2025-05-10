import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L245
class ViTPatchEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.img_size = cfg.vit_img_size
        self.patch_size = cfg.vit_patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.cls_flag = cfg.vit_cls_flag
        self.embd_dim = cfg.vit_hidden_dim

        # Conv layer to extract the patches
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.embd_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        if self.cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embd_dim))
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches + 1, self.embd_dim))
        else:
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches, self.embd_dim))


    def forward(self, x):
        x = self.conv(x)  # extract patches
        x = x.flatten(2)  # flatten the patches into a single dimension
        x = x.transpose(1, 2)  # transpose to (batch_size, num_patches, hidden_dim)

        # Add CLS token (according to original ViT Paper) and position embeddings
        if self.cls_flag:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = x + self.position_embedding
        return x

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L381
# https://github.com/karpathy/nanoGPT/blob/master/model.py#L29
class ViTMultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_heads = cfg.vit_n_heads
        self.embd_dim = cfg.vit_hidden_dim
        assert self.embd_dim % self.n_heads == 0, "embd_dim must be divisible by num_heads"
        self.head_dim = self.embd_dim // self.n_heads
        self.dropout = cfg.vit_dropout

        # Combined projections for all heads
        self.qkv_proj = nn.Linear(self.embd_dim, 3 * self.embd_dim, bias=True)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=True)

        # Dropout layers
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Use scaled dot product attention if available
        self.sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.sdpa:
            print("Warning: scaled dot product attention not available. Using standard attention in ViT.")

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        # Reshape  [B, T, C] -> [B, T, n_heads, head_dim] and transpose -> [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)

        if self.sdpa:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False # ViT attention is bidirectional
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v  # (B, n_heads, T, T) x (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        
        # Transpose back from [B, n_heads, T, head_dim] to [B, T, n_heads * head_dim] and combine all heads to [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L453
class ViTMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.activation_fn = nn.GELU(approximate='tanh')
        self.fc1 = nn.Linear(cfg.vit_hidden_dim, cfg.vit_inter_dim)
        self.fc2 = nn.Linear(cfg.vit_inter_dim, cfg.vit_hidden_dim)
        self.dropout = nn.Dropout(cfg.vit_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# https://github.com/karpathy/nanoGPT/blob/master/model.py#L94    
class ViTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.attn = ViTMultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.mlp = ViTMLP(cfg)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    

class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = ViTPatchEmbeddings(cfg)
        self.cls_flag = cfg.vit_cls_flag
        self.dropout = nn.Dropout(cfg.vit_dropout)
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.vit_n_blocks)])
        self.layer_norm = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.patch_embedding(x) 
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)

        if self.cls_flag:
            x = self.layer_norm(x[:, 0])
        else:
            x = self.layer_norm(x)
            #x = x.mean(dim=1)
        
        return x
    
    # Load the model from a pretrained HuggingFace model (we don't want to have to train the Vision Backbone from scratch)
    @classmethod
    def from_pretrained(cls, cfg):
        from transformers import SiglipVisionConfig
        from huggingface_hub import hf_hub_download
        import safetensors

        hf_config = SiglipVisionConfig.from_pretrained(cfg.vit_model_type)
        cfg.vit_dropout=hf_config.attention_dropout
        cfg.vit_hidden_dim=hf_config.hidden_size
        cfg.vit_img_size=hf_config.image_size
        cfg.vit_inter_dim=hf_config.intermediate_size
        cfg.vit_ln_eps=hf_config.layer_norm_eps
        cfg.vit_n_heads=hf_config.num_attention_heads
        cfg.vit_n_blocks=hf_config.num_hidden_layers
        cfg.vit_patch_size=hf_config.patch_size
        model = cls(cfg)
        safetensors_file = hf_hub_download(repo_id=cfg.vit_model_type, filename="model.safetensors")

        sd = model.state_dict()
        
        mapping = {
            'vision_model.embeddings.patch_embedding.weight': 'patch_embedding.conv.weight',
            'vision_model.embeddings.patch_embedding.bias': 'patch_embedding.conv.bias',
            'vision_model.embeddings.position_embedding.weight': 'patch_embedding.position_embedding',
            'vision_model.post_layernorm.weight': 'layer_norm.weight',
            'vision_model.post_layernorm.bias': 'layer_norm.bias',
        }
        
        for i in range(cfg.vit_n_blocks):
            # Layer norms
            mapping[f'vision_model.encoder.layers.{i}.layer_norm1.weight'] = f'blocks.{i}.ln1.weight'
            mapping[f'vision_model.encoder.layers.{i}.layer_norm1.bias'] = f'blocks.{i}.ln1.bias'
            mapping[f'vision_model.encoder.layers.{i}.layer_norm2.weight'] = f'blocks.{i}.ln2.weight'
            mapping[f'vision_model.encoder.layers.{i}.layer_norm2.bias'] = f'blocks.{i}.ln2.bias'
            
            # MLP
            mapping[f'vision_model.encoder.layers.{i}.mlp.fc1.weight'] = f'blocks.{i}.mlp.fc1.weight'
            mapping[f'vision_model.encoder.layers.{i}.mlp.fc1.bias'] = f'blocks.{i}.mlp.fc1.bias'
            mapping[f'vision_model.encoder.layers.{i}.mlp.fc2.weight'] = f'blocks.{i}.mlp.fc2.weight'
            mapping[f'vision_model.encoder.layers.{i}.mlp.fc2.bias'] = f'blocks.{i}.mlp.fc2.bias'
            
            # Output projection
            mapping[f'vision_model.encoder.layers.{i}.self_attn.out_proj.weight'] = f'blocks.{i}.attn.out_proj.weight'
            mapping[f'vision_model.encoder.layers.{i}.self_attn.out_proj.bias'] = f'blocks.{i}.attn.out_proj.bias'
        
        with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
            for hf_key, our_key in mapping.items():
                if hf_key in f.keys() and our_key in sd:
                    tensor = f.get_tensor(hf_key)
                    if tensor.shape == sd[our_key].shape:
                        sd[our_key].copy_(tensor)
                    else:
                        if 'position_embedding' in hf_key:
                            sd[our_key].copy_(tensor.unsqueeze(0))
                        else:
                            print(f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}")
                else:
                    if hf_key not in f.keys():
                        print(f"Warning: Key {hf_key} not found in safetensors file")
                    if our_key not in sd:
                        print(f"Warning: Key {our_key} not found in model state dict")
            
            # Manually handle QKV concatenation since our implementation combines Q, K, V into one
            for i in range(model.cfg.vit_n_blocks):
                q_weight = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.q_proj.weight')
                k_weight = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.k_proj.weight')
                v_weight = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.v_proj.weight')
                
                qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
                sd[f'blocks.{i}.attn.qkv_proj.weight'].copy_(qkv_weight)
                
                q_bias = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.q_proj.bias')
                k_bias = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.k_proj.bias')
                v_bias = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.v_proj.bias')
                
                qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
                sd[f'blocks.{i}.attn.qkv_proj.bias'].copy_(qkv_bias)
        
        model.load_state_dict(sd)
        print(f"Successfully loaded {cfg.vit_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
        return model
