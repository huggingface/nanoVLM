import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional


from models.utils import top_k_top_p_filtering
from models.vision_transformer import ViT
from models.language_model import LanguageModel
from models.modality_projector import ModalityProjector
from models.config import VLMConfig

from data.processors import get_tokenizer

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
        self.tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens)

    def forward(self, input_ids, image, attention_mask=None, targets=None):
        image_embd = self.vision_encoder(image)
        image_embd = self.MP(image_embd) # [B, IMAGE_TOKEN_LENGTH, D_lm]

        token_embd = self.decoder.token_embedding(input_ids) # [B, T_sequence, D_lm]

        # Identify image token placeholder positions and replace their embeddings
        # and cfg.IMAGE_TOKEN_LENGTH is the number of image tokens to replace.
        image_token_id = self.tokenizer.image_token_id 
        
        # For each item in the batch, find the start of the image token sequence
        # We expect cfg.IMAGE_TOKEN_LENGTH consecutive image_token_id
        # We also expect a BOI and EOI token, but we only replace the image_token_id ones.
        # The collator should ensure input_ids has self.cfg.IMAGE_TOKEN_LENGTH placeholders
        
        updated_token_embd = token_embd.clone()

        # Vectorized replacement of image token placeholders.
        # This assumes that `input_ids` contains `self.cfg.IMAGE_TOKEN_LENGTH` placeholders
        # for image tokens, and `image_embd` has the corresponding features.
        # The fallback logic for mismatched token counts is omitted as per request.
        
        batch_size = input_ids.size(0)
        image_token_len = self.cfg.IMAGE_TOKEN_LENGTH # Renamed L to image_token_len for clarity

        # Find the start index of image tokens for each item in the batch.
        # `torch.argmax` on the boolean mask (cast to int) returns the first index of `True` (value 1).
        # This relies on image_token_id being present and marking the start of the block.
        start_indices = torch.argmax((input_ids == image_token_id).int(), dim=1) # Shape: [batch_size]

        # Create batch indices for advanced indexing.
        # This tensor will be like [[0,0,..0], [1,1,..1], ..., [B-1,B-1,..B-1]]
        # where each inner list has length `image_token_len`.
        batch_idx_fill = torch.arange(batch_size, device=input_ids.device).unsqueeze(1).expand(-1, image_token_len) # Shape: [batch_size, image_token_len]

        # Create sequence indices for replacement.
        # `seq_offsets` will be [0, 1, ..., image_token_len-1]
        seq_offsets = torch.arange(image_token_len, device=input_ids.device).unsqueeze(0) # Shape: [1, image_token_len]
        # `start_indices.unsqueeze(1)` results in shape [batch_size, 1].
        # Broadcasting with `seq_offsets` gives, for each batch item `b`:
        # [start_indices[b], start_indices[b]+1, ..., start_indices[b]+image_token_len-1]
        sequence_idx_fill = start_indices.unsqueeze(1) + seq_offsets # Shape: [batch_size, image_token_len]
        
        # Perform the replacement using advanced indexing.
        # `updated_token_embd[batch_idx_fill, sequence_idx_fill]` selects slices of shape [batch_size, image_token_len, D_lm]
        # `image_embd` also has shape [batch_size, image_token_len, D_lm] (or [B, IMAGE_TOKEN_LENGTH, D_lm])
        updated_token_embd[batch_idx_fill, sequence_idx_fill] = image_embd.to(updated_token_embd.dtype)

        # The combined_embd is now the token_embd with image parts replaced.
        # The attention_mask comes from the collator and should already cover the full sequence.
        logits, _ = self.decoder(updated_token_embd, attention_mask=attention_mask)

        loss = None
        if targets is not None:
            logits = self.decoder.head(logits) # Apply LM head
            # Loss is calculated over all tokens, but `targets` (labels) will have -100 for non-answer tokens.
            # No need to slice logits based on image embedding size here, as the target mask handles it.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        return logits, loss

    @torch.inference_mode()
    def generate(self, input_ids, image, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):

        # 1. Process image
        image_embd = self.vision_encoder(image) # [B, T_img_feat, D_model]
        image_embd = self.MP(image_embd)      # [B, IMAGE_TOKEN_LENGTH, D_lm]

        # 2. Embed initial text prompt tokens
        prompt_token_embeds = self.decoder.token_embedding(input_ids) # [B, T_prompt_text, D_lm]
        
        # Replace image token placeholders in prompt_token_embeds with image_embd
        image_token_id = self.tokenizer.image_token_id
        
        initial_combined_embeds = prompt_token_embeds.clone()

        # Vectorized replacement of image token placeholders.
        batch_size = input_ids.size(0)
        image_token_len = self.cfg.IMAGE_TOKEN_LENGTH

        # Find the start index of image tokens for each item in the batch.
        start_indices = torch.argmax((input_ids == image_token_id).int(), dim=1)

        # Create batch indices for advanced indexing.
        batch_idx_fill = torch.arange(batch_size, device=input_ids.device).unsqueeze(1).expand(-1, image_token_len)

        # Create sequence indices for replacement.
        seq_offsets = torch.arange(image_token_len, device=input_ids.device).unsqueeze(0)
        sequence_idx_fill = start_indices.unsqueeze(1) + seq_offsets
        
        # Perform the replacement using advanced indexing.
        initial_combined_embeds[batch_idx_fill, sequence_idx_fill] = image_embd.to(initial_combined_embeds.dtype)

        current_total_seq_len = initial_combined_embeds.size(1)
        batch_size = image_embd.size(0) # Or initial_combined_embeds.size(0)
        
        # The attention_mask from input should correspond to input_ids and already be correct.
        # No explicit concatenation needed here if collator prepares it correctly.

        # --- Multimodal Prefill Phase ---
        prefill_output, kv_cache_list = self.decoder(
            initial_combined_embeds,
            attention_mask=attention_mask, # Use the provided attention mask
            kv_cache=None,
            start_pos=0
        )
        
        last_token_output_from_prefill = prefill_output[:, -1, :] 
        
        if not self.decoder.lm_use_tokens:
            current_logits = self.decoder.head(last_token_output_from_prefill) 
        else:
            current_logits = last_token_output_from_prefill 

        # Store newly generated token IDs
        newly_generated_ids_list = []

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
            current_token_start_pos = current_total_seq_len
            current_total_seq_len += 1

            # update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)

            # With KV cache: only process the new token
            decode_step_output, kv_cache_list = self.decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
            )
      
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
