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
        # Encode image and text separately
        image_embd = self.vision_encoder(image)
        image_embd = self.MP(image_embd)
        token_embd = self.decoder.token_embedding(input_ids)

        # Concatenate image embeddings to token embeddings
        combined_embd = torch.cat((image_embd, token_embd), dim=1)
        
        # Adjust attention mask to account for image tokens
        if attention_mask is not None:
            batch_size = image_embd.size(0)
            img_seq_len = image_embd.size(1)
            image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)

        # Pass through decoder (returns hidden states if lm_use_tokens=False or logits if True)
        output_from_decoder = self.decoder(combined_embd, attention_mask=attention_mask, use_cache=False) 

        loss = None
        if targets is not None:
            # Apply head to get logits if decoder returned hidden states
            logits = output_from_decoder
            if not self.decoder.lm_use_tokens:
                logits = self.decoder.head(logits)
            
            # Use only the token part for loss computation
            logits = logits[:, image_embd.size(1):, :]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        
        return output_from_decoder, loss

    @torch.no_grad()
    def generate(self, input_ids, image, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):
        # Process image through vision encoder and projection
        image_embd = self.vision_encoder(image)
        image_embd = self.MP(image_embd)
        
        # Embed initial tokens
        token_embd = self.decoder.token_embedding(input_ids)
        
        # Concatenate image embeddings with token embeddings
        initial_embeddings = torch.cat((image_embd, token_embd), dim=1)

        batch_size = initial_embeddings.size(0)
        prompt_len = initial_embeddings.size(1)
        
        # Prepare attention mask for the initial prompt processing
        prompt_attention_mask = None
        if attention_mask is not None:
            image_attention_mask = torch.ones((batch_size, image_embd.size(1)), device=attention_mask.device, dtype=attention_mask.dtype)
            prompt_attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)
        
        # Initialize KV cache for each block in the decoder
        kv_caches = [None] * len(self.decoder.blocks)
        
        # Process initial prompt sequence to populate the KV cache
        prompt_pos_ids = torch.arange(prompt_len, device=initial_embeddings.device).unsqueeze(0).expand(batch_size, -1)
        prompt_cos, prompt_sin = self.decoder.rotary_embd(prompt_pos_ids)
        
        # Pass initial embeddings through all blocks to get hidden states
        current_hidden_states = initial_embeddings
        for i, block in enumerate(self.decoder.blocks):
            current_hidden_states, kv_caches[i] = block(current_hidden_states, prompt_cos, prompt_sin, prompt_attention_mask, None, True)
        
        # Get final hidden state of the prompt for the first token prediction
        prompt_last_hidden = self.decoder.norm(current_hidden_states[:, -1, :])  # [B, C]

        # Initialize tensor to store generated token IDs
        generated_tokens = torch.zeros((batch_size, max_new_tokens), device=input_ids.device, dtype=input_ids.dtype)
        
        # Store embeddings between generation steps
        next_token_embeddings = None

        for k in range(max_new_tokens):
            if k == 0:
                # For first new token, use last hidden state from prompt
                current_logits = self.decoder.head(prompt_last_hidden)
            else:
                # For subsequent tokens, use embedding of previous token
                current_input = next_token_embeddings  # [B, 1, C]
                
                # Get position for the current token
                pos_id = torch.tensor([[prompt_len + k - 1]], device=initial_embeddings.device).expand(batch_size, -1)
                cos, sin = self.decoder.rotary_embd(pos_id)
                
                # Process current token through blocks with KV cache
                hidden_state = current_input
                for i, block in enumerate(self.decoder.blocks):
                    hidden_state, kv_caches[i] = block(hidden_state, cos, sin, None, kv_caches[i], True)
                
                # Get normalized hidden state
                hidden_state_normed = self.decoder.norm(hidden_state)
                current_logits = self.decoder.head(hidden_state_normed.squeeze(1))

            # Sample next token
            probs = F.softmax(current_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, k] = next_token.squeeze(-1)
            
            # Get embedding for next iteration
            next_token_embeddings = self.decoder.token_embedding(next_token)
            
        return generated_tokens

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
