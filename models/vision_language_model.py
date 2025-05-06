from models.vision_transformer import ViT
from models.language_model import LanguageModel
from models.modality_projector import ModalityProjector

import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionLanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = ViT(cfg)
        self.decoder = LanguageModel(cfg)
        self.MP = ModalityProjector(cfg)

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
    def generate(self, input_ids, image, attention_mask=None, max_new_tokens=5):
        # Process image through vision encoder and projection
        image_embd = self.vision_encoder(image)
        image_embd = self.MP(image_embd)
        
        # Embed initial tokens
        token_embd = self.decoder.token_embedding(input_ids)
        
        # Concatenate image embeddings with token embeddings
        combined_embd = torch.cat((image_embd, token_embd), dim=1)

        batch_size = image_embd.size(0)
        img_seq_len = image_embd.size(1)
        # Adjust attention mask to account for image tokens
        if attention_mask is not None:
            # Create mask of 1s for image tokens (all image tokens should be attended to)
            image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)
        
        # Generate from combined embeddings using the decoder
        # We need to use the decoder's forward function and not its generate method
        # because we want to keep track of the image prefix
        outputs = combined_embd
        generated_tokens = torch.zeros((batch_size, max_new_tokens), device=input_ids.device, dtype=input_ids.dtype)
        
        #Note: Here you could implement improvements like e.g. KV caching
        for i in range(max_new_tokens):
            model_out = self.decoder(outputs, attention_mask)
            
            # Get predictions for the last token only (normally this is the embedding, not the logits)
            last_token_logits = model_out[:, -1, :]
            
            # Apply head to get logits (if model is in embedding mode)
            if not self.decoder.lm_use_tokens:
                last_token_logits = self.decoder.head(last_token_logits)

            probs = torch.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
                
            generated_tokens[:, i] = next_token.squeeze(-1)
            
            # Convert to embedding and append
            next_embd = self.decoder.token_embedding(next_token)
            outputs = torch.cat((outputs, next_embd), dim=1)

            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device)), dim=1)
        
        return generated_tokens
        
    def load_checkpoint(self, path: str):
        print(f"Loading weights from full VLM checkpoint: {path}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if path.lower().endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError:
                raise ImportError("safetensors is not installed. Please install it with `pip install safetensors` to load .safetensors files.")
            checkpoint = load_file(path, device=str(device))
        elif path.lower().endswith(".pth") or path.lower().endswith(".pt"): # Common PyTorch extensions
            checkpoint = torch.load(path, map_location=device)
        else:
            print(f"Warning: Unknown file extension for checkpoint: {path}. Attempting to load with torch.load.")
            checkpoint = torch.load(path, map_location=device)
            
        self.load_state_dict(checkpoint)

    @classmethod
    def from_pretrained(cls, cfg):
        model = cls(cfg)
        model.vision_encoder = ViT.from_pretrained(cfg)
        model.decoder = LanguageModel.from_pretrained(cfg)

        return model
