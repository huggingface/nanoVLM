import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional, Literal


from models.kvcache import KVCacheFactory
from models.utils import top_k_top_p_filtering
from models.vision_transformer import ViT
from models.language_model import LanguageModel
from models.modality_projector import ModalityProjector
from models.config import VLMConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LanguageModel(cfg)
        self.MP = ModalityProjector(cfg)
        self.load_backbone = load_backbone

    def forward(self, input_ids, image, attention_mask=None, targets=None):
        image_embd = self.vision_encoder(image)
        image_embd = self.MP(image_embd)

        token_embd = self.decoder.token_embedding(input_ids)

        combined_embd = torch.cat((image_embd, token_embd), dim=1) # Concatenate image embeddings to token embeddings

        # Adjust attention mask to account for image tokens
        if attention_mask is not None:
            # Create mask of 1s for image tokens (all image tokens should be attended to)
            batch_size = image_embd.size(0)
            img_seq_len = image_embd.size(1)
            image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)
   
            # Combine image and token attention masks
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)

        logits, _ = self.decoder(combined_embd, attention_mask=attention_mask) # Not logits yet, but easier to return like this

        loss = None
        if targets is not None:
            # Only use the token part of the logits for loss computation
            logits = self.decoder.head(logits)
            logits = logits[:, image_embd.size(1):, :]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        return logits, loss

    def setup_kv_cache(self, batch_size, max_seq_length, kv_cache_implementation: Literal["static", "dynamic"] = "static"):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        embd_dim = self.cfg.lm_hidden_dim
        n_heads = self.cfg.lm_n_heads

        head_dim = embd_dim // n_heads

        kv_caches = [
            KVCacheFactory.create_kv_cache(
                kv_cache_implementation,
                batch_size,
                max_seq_length,
                self.cfg.lm_n_kv_heads,
                head_dim,
                dtype=dtype,
            ).to(device) for _ in range(self.cfg.lm_n_blocks)
        ]
        causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool, device=device))

        return kv_caches, causal_mask

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        image,
        attention_mask=None,
        max_new_tokens=5,
        top_k=50,
        top_p=0.9,
        temperature=0.5,
        greedy=False,
        use_kv_cache=False,
        kv_cache_implementation: Optional[Literal["static", "dynamic"]] = "static",
    ):

        # 1. Process image
        image_embd = self.vision_encoder(image) # [B, T_img, D_model]
        image_embd = self.MP(image_embd)      # [B, T_img, D_lm]

        # 2. Embed initial text prompt tokens
        prompt_token_embeds = self.decoder.token_embedding(input_ids) # [B, T_prompt_text, D_lm]

        # 3. Combine image and text prompt embeddings for prefill
        initial_combined_embeds = torch.cat((image_embd, prompt_token_embeds), dim=1) # [B, T_img + T_prompt_text, D_lm]
        current_total_seq_len = initial_combined_embeds.size(1)

        batch_size = image_embd.size(0)
        if attention_mask is not None:
            # Create mask of 1s for image tokens (all image tokens should be attended to)
            img_seq_len = image_embd.size(1)
            image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)

            # Combine image and token attention masks
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)

        # print(f"attention_mask: {attention_mask}")
        current_position_ids = torch.arange(current_total_seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        kv_caches, casual_mask = self.setup_kv_cache(
            batch_size,
            initial_combined_embeds.size(-1) + max_new_tokens,
            kv_cache_implementation=kv_cache_implementation,
        )

        if use_kv_cache:
            if kv_cache_implementation == "static":
                attention_mask = casual_mask[None, None, current_position_ids[0]]
            elif kv_cache_implementation == "dynamic":
                attention_mask = casual_mask[None, None, current_position_ids[0]][..., :current_total_seq_len]
            else:
                attention_mask = attention_mask

        # --- Multimodal Prefill Phase ---
        prefill_output = self.decoder(
            initial_combined_embeds,
            attention_mask=attention_mask,
            kv_cache=kv_caches,
            current_position_ids=current_position_ids
        )

        last_token_output_from_prefill = prefill_output[:, -1, :] 

        if not self.decoder.lm_use_tokens:
            current_logits = self.decoder.head(last_token_output_from_prefill) 
        else:
            current_logits = last_token_output_from_prefill 

        # Store newly generated token IDs
        newly_generated_ids_list = []
        current_token_start_pos = torch.tensor([current_total_seq_len + 1], device=input_ids.device).expand(batch_size, -1) - 1

        # --- Decode Phase by sampling tokens autoregressively using the kv-cache ---
        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            newly_generated_ids_list.append(next_token_id)
            
            # Embed the newly generated token
            next_token_embed = self.decoder.token_embedding(next_token_id) # [B, 1, D_lm]
            
            # The start_pos for the new token is the current total sequence length *before* adding this new token
            current_total_seq_len += 1

            # update attention mask
            if use_kv_cache:
                if kv_cache_implementation == "static":
                    attention_mask = casual_mask[None, None, current_token_start_pos[0]]
                elif kv_cache_implementation == "dynamic":
                    attention_mask = casual_mask[None, None, current_token_start_pos[0]][..., :current_total_seq_len]
                else:
                    attention_mask = None

            # With KV cache: only process the new token
            decode_step_output = self.decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_caches,
                current_position_ids=current_token_start_pos
            )

            current_token_start_pos += 1

            last_token_output = decode_step_output[:, -1, :] 

            # Apply head to get logits (if model is in embedding mode)
            if not self.decoder.lm_use_tokens:
                current_logits = self.decoder.head(last_token_output)
            else:
                current_logits = last_token_output

        if not newly_generated_ids_list: # Handle case where max_new_tokens might be 0
            return torch.empty((batch_size,0), dtype=torch.long, device=input_ids.device)

        return torch.cat(newly_generated_ids_list, dim=1)

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "VisionLanguageModel":
        """
        Load a VisionLanguageModel from a local directory or a repo on the Hugging Face Hub.

        Args:
            repo_id_or_path (str): The path to the local directory or the Hugging Face Hub repo ID.

        Returns:
            VisionLanguageModel: The loaded model.
        """
        # If local folder exists => load from there
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path."
                )
        # Otherwise, assume it's a Hugging Face Hub repo
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        # Load config
        with open(config_path, "r") as f:
            cfg = VLMConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(cfg, load_backbone=False)

        # Load safetensors weights
        load_model(model, weights_path)

        # Done!
        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and configuration to a directory.

        Args:
            save_directory (str): The directory to save the model and config.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        # Save weights as safetensors
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        """
        Push the model and configuration to the Hugging Face Hub.

        Args:
            repo_id (str): The repo ID on the Hugging Face Hub.
        """
        from huggingface_hub import create_repo, upload_folder

        # Create repo
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            # Save to tmp directory
            self.save_pretrained(save_path)

            # Save model card
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            # Upload
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload nanoVLM using push_to_hub",
            )


MODEL_CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
library_name: nanovlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - research
---

**nanoVLM** is a minimal and lightweight Vision-Language Model (VLM) designed for efficient training and experimentation. Built using pure PyTorch, the entire model architecture and training logic fits within ~750 lines of code. It combines a ViT-based image encoder (SigLIP-B/16-224-85M) with a lightweight causal language model (SmolLM2-135M), resulting in a compact 222M parameter model.

For more information, check out the base model on https://huggingface.co/lusxvr/nanoVLM-222M.

**Usage:**

Clone the nanoVLM repository: https://github.com/huggingface/nanoVLM.
Follow the install instructions and run the following code:

```python
from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("{repo_id}")
```
"""
