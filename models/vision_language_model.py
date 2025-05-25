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

        logits = self.decoder(combined_embd, attention_mask) # Not logits yet, but easier to return like this

        loss = None
        if targets is not None:
            # Only use the token part of the logits for loss computation
            logits = self.decoder.head(logits)
            logits = logits[:, image_embd.size(1):, :]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def generate(self, 
                 input_ids, 
                 image, 
                 attention_mask=None, 
                 max_new_tokens=5, 
                 top_k=50, 
                 top_p=0.9, 
                 temperature=0.5, 
                 greedy=False, 
                 beam_size: int = 1,
                 length_penalty: float = 1.0):
        # process image through vision encoder and projection
        image_embd = self.vision_encoder(image)
        image_embd = self.MP(image_embd)
        
        #branch to top-k beam search if requested from generate.py
        if beam_size > 1:
            return self._beam_search(
                input_ids,
                image_embd,
                attention_mask,
                max_new_tokens,
                beam_size,
                length_penalty,
                temperature
            )

        # embed the initial text tokens
        token_embd = self.decoder.token_embedding(input_ids)
        # concatenate image and token embeddings along sequence dimension
        outputs = torch.cat((image_embd, token_embd), dim=1)

        if attention_mask is not None:
            batch_size, img_seq_len = image_embd.shape[:2]
            image_mask = torch.ones(
                (batch_size, img_seq_len),
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            attention_mask = torch.cat((image_mask, attention_mask), dim=1)

        batch_size = image_embd.size(0)
        generated = torch.zeros(
            (batch_size, max_new_tokens),
            device=input_ids.device,
            dtype=input_ids.dtype
        )

        for i in range(max_new_tokens):
            model_out = self.decoder(outputs, attention_mask)
            # take logits for the last time step
            last_logits = model_out[:, -1, :]
            # if decoder returns embeddings instead of logits, apply the head
            if not self.decoder.lm_use_tokens:
                last_logits = self.decoder.head(last_logits)

            if greedy:
                # greedy search just picks highest‐probability token...
                next_tok = torch.argmax(last_logits, dim=-1, keepdim=True)
            else:
                # apply top-k/top-p filtering then sample
                filt = top_k_top_p_filtering(last_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filt / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            #record the chosen token
            generated[:, i] = next_tok.squeeze(-1)
            next_embd = self.decoder.token_embedding(next_tok)
            outputs = torch.cat((outputs, next_embd), dim=1)

            # extend attention mask for the new token if present
            if attention_mask is not None:
                pad = torch.ones((batch_size, 1), device=attention_mask.device)
                attention_mask = torch.cat((attention_mask, pad), dim=1)

        return generated
    
    def _beam_search(
        self,
        input_ids,
        image_embd,
        attention_mask,
        max_new_tokens,
        beam_size,
        length_penalty,
        temperature
    ):
        batch_size = image_embd.size(0)
        #(sequence_ids, accumulated_score, attention_mask)
        beams = [(input_ids, 0.0, attention_mask)]
        completed = []

        for _ in range(max_new_tokens):
            candidates = []
            for seq, score, mask in beams:
                #single‐step forward
                tok_emb = self.decoder.token_embedding(seq)
                combined = torch.cat((image_embd, tok_emb), dim=1)
                logits = self.decoder(combined, mask)[:, -1, :]
                if not self.decoder.lm_use_tokens:
                    logits = self.decoder.head(logits)
                log_probs = F.log_softmax(logits / temperature, dim=-1)

                # top‐k expansions from this beam
                topk_lp, topk_ids = log_probs.topk(beam_size, dim=-1)
                for lp, tid in zip(topk_lp[0], topk_ids[0]):
                    # i had a shape issue here so I had to unsqueeze twice to solve it
                    # unsqueeze twice so we get a (batch_size, 1) tensor
                    new_tok = tid.unsqueeze(0).unsqueeze(-1)
                    #seq (shape [batch, seq_len]) and new_tok ([batch,1]) are now 2-D
                    new_seq = torch.cat([seq, new_tok], dim=1)
                    if mask is not None:
                        new_mask = torch.cat(
                            [mask, torch.ones((batch_size, 1), device=mask.device)],
                            dim=1
                        )
                    else:
                        new_mask = None
                    candidates.append((new_seq, score + lp.item(), new_mask))

            # cut back to best `beam_size` (with length penalty)
            beams = sorted(
                candidates,
                key=lambda x: x[1] / (x[0].size(1) ** length_penalty),
                reverse=True
            )[:beam_size]

            if not beams:
                break

        # pick the best finished or ongoing beam
        beam_seqs = [b[0][:, input_ids.size(1):] for b in beams]
        # return ONLY the newly generated portion
        return torch.stack(beam_seqs, dim=1)


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
