"""
ONNX Runtime inference script for nanoVLM.

This script provides an inference pipeline using the exported ONNX models.

Usage:
    python inference_onnx.py --onnx_dir onnx_models --image assets/image.png --prompt "What is this?"
"""

import argparse
import json
import os
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Tuple

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "onnxruntime is required for ONNX inference. "
        "Install it with: pip install onnxruntime or pip install onnxruntime-gpu"
    )

from data.processors import get_tokenizer, get_image_processor, get_image_string
from models.config import VLMConfig


@dataclass
class ONNXModelPaths:
    """Paths to ONNX model files."""
    vision_encoder: str
    modality_projector: str
    language_decoder_prefill: str
    language_decoder_decode: str
    config: str


class NanoVLMONNXInference:
    """ONNX Runtime inference for nanoVLM."""

    def __init__(self, onnx_dir: str, device: str = 'cpu'):
        """
        Initialize ONNX inference.

        Args:
            onnx_dir: Directory containing ONNX models
            device: Device to run on ('cpu' or 'cuda')
        """
        self.onnx_dir = onnx_dir
        self.device = device

        # Set up ONNX Runtime providers
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Load model paths
        self.paths = ONNXModelPaths(
            vision_encoder=os.path.join(onnx_dir, "vision_encoder.onnx"),
            modality_projector=os.path.join(onnx_dir, "modality_projector.onnx"),
            language_decoder_prefill=os.path.join(onnx_dir, "language_decoder_prefill.onnx"),
            language_decoder_decode=os.path.join(onnx_dir, "language_decoder_decode.onnx"),
            config=os.path.join(onnx_dir, "config.json"),
        )

        # Validate paths
        for path_name, path in self.paths.__dict__.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"ONNX model file not found: {path}")

        # Load config
        with open(self.paths.config, 'r') as f:
            config_dict = json.load(f)
        self.cfg = VLMConfig(**config_dict)

        # Initialize ONNX Runtime sessions
        print(f"Loading ONNX models on {device}...")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.vision_encoder_session = ort.InferenceSession(
            self.paths.vision_encoder, providers=providers, sess_options=sess_options
        )
        self.modality_projector_session = ort.InferenceSession(
            self.paths.modality_projector, providers=providers, sess_options=sess_options
        )
        self.decoder_prefill_session = ort.InferenceSession(
            self.paths.language_decoder_prefill, providers=providers, sess_options=sess_options
        )
        self.decoder_decode_session = ort.InferenceSession(
            self.paths.language_decoder_decode, providers=providers, sess_options=sess_options
        )

        # Load tokenizer and image processor
        self.tokenizer = get_tokenizer(
            self.cfg.lm_tokenizer,
            self.cfg.vlm_extra_tokens,
            self.cfg.lm_chat_template
        )
        self.image_processor = get_image_processor(
            self.cfg.max_img_size,
            self.cfg.vit_img_size,
            self.cfg.resize_to_max_side_len
        )

        # Get token embedding layer (needed for converting token IDs to embeddings)
        # We'll need to load this from the original model
        self._load_embedding_layer()

        print("ONNX models loaded successfully!")

    def _load_embedding_layer(self):
        """Load token embedding layer from the original model checkpoint."""
        # We need this to convert token IDs to embeddings for the decoder
        import torch
        from models.vision_language_model import VisionLanguageModel

        # Try to find original checkpoint
        original_checkpoint = self.onnx_dir.replace('onnx_models', '')
        if not original_checkpoint or not os.path.exists(os.path.join(original_checkpoint, 'model.safetensors')):
            # Default to HF model
            original_checkpoint = self.cfg.hf_repo_name if hasattr(self.cfg, 'hf_repo_name') else 'lusxvr/nanoVLM-450M'

        print(f"Loading embedding layer from {original_checkpoint}...")
        with torch.no_grad():
            model = VisionLanguageModel.from_pretrained(original_checkpoint)
            # Extract embeddings and LM head as numpy arrays
            self.token_embeddings = model.decoder.token_embedding.weight.cpu().numpy()
            self.lm_head_weight = model.decoder.head.weight.cpu().numpy()

        print(f"Loaded embeddings: {self.token_embeddings.shape}")

    def process_image(self, image_path: str) -> np.ndarray:
        """
        Process an image for the vision encoder.

        Args:
            image_path: Path to image file

        Returns:
            Processed image as numpy array [1, 3, H, W]
        """
        image = Image.open(image_path).convert('RGB')
        processed = self.image_processor(image)

        # processed is a list of tensors (global + splits) or a tuple (global, splits)
        # For now, we'll just use the first image (global view)
        if isinstance(processed, (list, tuple)):
            image_tensor = processed[0]  # Global image
        else:
            image_tensor = processed

        # Convert to numpy
        import torch
        if isinstance(image_tensor, torch.Tensor):
            image_np = image_tensor.numpy()
        else:
            image_np = np.array(image_tensor)

        # Add batch dimension if needed
        if len(image_np.shape) == 3:
            image_np = np.expand_dims(image_np, axis=0)
        elif len(image_np.shape) == 4 and image_np.shape[0] != 1:
            # If batch dimension exists but is not 1, take first item
            image_np = image_np[0:1]

        return image_np

    def encode_image(self, image_np: np.ndarray) -> np.ndarray:
        """
        Encode image to vision features.

        Args:
            image_np: Image array [batch_size, 3, H, W]

        Returns:
            Vision features [batch_size, num_patches, vit_hidden_dim]
        """
        vision_features = self.vision_encoder_session.run(
            None,
            {'images': image_np}
        )[0]
        return vision_features

    def project_vision_features(self, vision_features: np.ndarray) -> np.ndarray:
        """
        Project vision features to language space.

        Args:
            vision_features: [batch_size, num_patches, vit_hidden_dim]

        Returns:
            Projected features [batch_size, mp_image_token_length, lm_hidden_dim]
        """
        projected_features = self.modality_projector_session.run(
            None,
            {'vision_features': vision_features}
        )[0]
        return projected_features

    def prepare_inputs(
        self,
        image_path: str,
        prompt: str
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Prepare inputs for the model.

        Args:
            image_path: Path to image
            prompt: Text prompt

        Returns:
            Tuple of (embeddings, attention_mask, image_counts)
        """
        # Process image
        image_np = self.process_image(image_path)

        # Get image embeddings
        vision_features = self.encode_image(image_np)
        image_embeddings = self.project_vision_features(vision_features)

        # For now, assume single image (no splitting)
        # splitted_image_counts is a list of (height_patches, width_patches) tuples
        splitted_image_counts = [(1, 1)]

        # Create image string with special tokens
        image_string = get_image_string(self.tokenizer, splitted_image_counts, self.cfg.mp_image_token_length)

        # Format prompt with image
        messages = [{"role": "user", "content": image_string + prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        encoded = self.tokenizer(
            prompt_text,
            return_tensors='np',
            padding=False,
            truncation=True,
            max_length=self.cfg.lm_max_length
        )

        input_ids = encoded['input_ids']  # [1, seq_len]
        attention_mask = encoded['attention_mask']  # [1, seq_len]

        # Convert token IDs to embeddings
        token_embeddings = self.token_embeddings[input_ids[0]]  # [seq_len, hidden_dim]
        token_embeddings = np.expand_dims(token_embeddings, axis=0)  # [1, seq_len, hidden_dim]

        # Replace image token placeholders with image embeddings
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.image_token)
        combined_embeddings = self._replace_image_tokens(
            token_embeddings, input_ids, image_embeddings, image_token_id
        )

        return combined_embeddings, attention_mask, splitted_image_counts

    def _replace_image_tokens(
        self,
        token_embeddings: np.ndarray,
        input_ids: np.ndarray,
        image_embeddings: np.ndarray,
        image_token_id: int
    ) -> np.ndarray:
        """
        Replace image token placeholders with actual image embeddings.

        Args:
            token_embeddings: [batch_size, seq_len, hidden_dim]
            input_ids: [batch_size, seq_len]
            image_embeddings: [num_images, mp_image_token_length, hidden_dim]
            image_token_id: ID of the image token

        Returns:
            Combined embeddings [batch_size, seq_len, hidden_dim]
        """
        # Find positions of image tokens
        image_token_mask = (input_ids == image_token_id)

        # Flatten image embeddings
        image_emb_flat = image_embeddings.reshape(-1, image_embeddings.shape[-1])

        # Replace image tokens
        combined = token_embeddings.copy()
        image_idx = 0
        for batch_idx in range(input_ids.shape[0]):
            for seq_idx in range(input_ids.shape[1]):
                if image_token_mask[batch_idx, seq_idx]:
                    if image_idx < len(image_emb_flat):
                        combined[batch_idx, seq_idx] = image_emb_flat[image_idx]
                        image_idx += 1

        return combined

    def generate(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        greedy: bool = False
    ) -> str:
        """
        Generate text from image and prompt.

        Args:
            image_path: Path to image
            prompt: Text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            greedy: If True, use greedy decoding

        Returns:
            Generated text
        """
        # Prepare inputs
        embeddings, attention_mask, _ = self.prepare_inputs(image_path, prompt)

        # Check expected sequence length from model
        expected_seq_len = self.decoder_prefill_session.get_inputs()[0].shape[1]
        actual_seq_len = embeddings.shape[1]

        # Pad to expected length if needed
        if actual_seq_len < expected_seq_len:
            pad_len = expected_seq_len - actual_seq_len
            embeddings = np.pad(embeddings, ((0, 0), (0, pad_len), (0, 0)), mode='constant')
            attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_len)), mode='constant')
        elif actual_seq_len > expected_seq_len:
            # Truncate if too long
            embeddings = embeddings[:, :expected_seq_len, :]
            attention_mask = attention_mask[:, :expected_seq_len]

        # Prefill phase
        prefill_inputs = {
            'embeddings': embeddings.astype(np.float32),
            'attention_mask': attention_mask.astype(np.int64)
        }

        prefill_outputs = self.decoder_prefill_session.run(None, prefill_inputs)
        hidden_states = prefill_outputs[0]  # [batch_size, seq_len, hidden_dim]
        kv_cache = prefill_outputs[1:]  # List of KV cache tensors

        # Get logits for last token
        last_hidden = hidden_states[:, -1:, :]  # [batch_size, 1, hidden_dim]
        logits = np.dot(last_hidden[0, 0], self.lm_head_weight.T)  # [vocab_size]

        # Sample first token
        next_token_id = self._sample_token(logits, temperature, top_k, top_p, greedy)
        generated_ids = [next_token_id]

        # Decode phase
        current_seq_len = embeddings.shape[1]

        for step in range(max_new_tokens - 1):
            # Get embedding for next token
            next_token_embedding = self.token_embeddings[next_token_id]  # [hidden_dim]
            next_token_embedding = np.expand_dims(next_token_embedding, axis=(0, 1))  # [1, 1, hidden_dim]

            # Update attention mask
            attention_mask = np.concatenate([
                attention_mask,
                np.ones((1, 1), dtype=np.int64)
            ], axis=1)

            # Prepare decode inputs
            decode_inputs = {
                'embeddings': next_token_embedding.astype(np.float32),
                'attention_mask': attention_mask.astype(np.int64),
                'start_pos': np.array([current_seq_len], dtype=np.int64),
            }

            # Add KV cache to inputs
            # Map prefill outputs to decode inputs using actual input names
            decode_input_names = [inp.name for inp in self.decoder_decode_session.get_inputs()]
            kv_input_names = [name for name in decode_input_names if name.startswith('kv_cache')]

            for i, kv_input_name in enumerate(kv_input_names):
                if i < len(kv_cache):
                    decode_inputs[kv_input_name] = kv_cache[i].astype(np.float32)

            # Run decode
            decode_outputs = self.decoder_decode_session.run(None, decode_inputs)
            hidden_states = decode_outputs[0]  # [batch_size, 1, hidden_dim]
            kv_cache = decode_outputs[1:]  # Updated KV cache

            # Get logits
            logits = np.dot(hidden_states[0, 0], self.lm_head_weight.T)  # [vocab_size]

            # Sample next token
            next_token_id = self._sample_token(logits, temperature, top_k, top_p, greedy)

            # Check for EOS
            if next_token_id == self.tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id)
            current_seq_len += 1

        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text

    def _sample_token(
        self,
        logits: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
        greedy: bool
    ) -> int:
        """
        Sample next token from logits.

        Args:
            logits: Logits array [vocab_size]
            temperature: Sampling temperature
            top_k: Top-k parameter
            top_p: Top-p parameter
            greedy: Use greedy decoding

        Returns:
            Sampled token ID
        """
        if greedy:
            return int(np.argmax(logits))

        # Apply temperature
        logits = logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
            logits[indices_to_remove] = -float('inf')

        # Convert to probabilities
        probs = self._softmax(logits)

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above threshold
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0.0

            # Renormalize
            probs = probs / probs.sum()

        # Sample
        token_id = np.random.choice(len(probs), p=probs)
        return int(token_id)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


def main():
    parser = argparse.ArgumentParser(description='Run ONNX inference for nanoVLM')
    parser.add_argument(
        '--onnx_dir',
        type=str,
        default='onnx_models',
        help='Directory containing ONNX models'
    )
    parser.add_argument(
        '--image',
        type=str,
        default='assets/image.png',
        help='Path to input image'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='What is this?',
        help='Text prompt'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=50,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Top-k sampling parameter'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p (nucleus) sampling parameter'
    )
    parser.add_argument(
        '--greedy',
        action='store_true',
        help='Use greedy decoding'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run on'
    )

    args = parser.parse_args()

    # Initialize inference
    inference = NanoVLMONNXInference(args.onnx_dir, device=args.device)

    # Generate
    print(f"\nInput image: {args.image}")
    print(f"Prompt: {args.prompt}")
    print("\nGenerating...")

    generated_text = inference.generate(
        image_path=args.image,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy
    )

    print(f"\nGenerated text:\n{generated_text}")


if __name__ == '__main__':
    main()
