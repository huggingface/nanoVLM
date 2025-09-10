from data.processors import get_tokenizer
from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig
import re
import json

import torch
from PIL import Image
import os

from transformers import (
    Idefics3Config,
    Idefics3ForConditionalGeneration,
    Idefics3ImageProcessor,
    Idefics3Processor,
)


def load_model_and_tokenizer(path_model):
    # Load tokenizer
    with open(path_model + "/config.json", "r") as f:
        config = VLMConfig(**json.load(f))
    tokenizer = get_tokenizer(config.lm_tokenizer)
    model = VisionLanguageModel.from_pretrained(path_model)
    # Eval mode
    model.eval()
    return tokenizer, model, config

tokenizer, nano_model, config = load_model_and_tokenizer("checkpoints/nanoVLM_siglip2-base-patch16-256_mp2_SmolLM2-360M-Instruct_8xGPU_1630251samples_bs1024_ep5_lr5e-05-0.005_0522_train_full")
# tokenizer, nano_model, config = load_model_and_tokenizer("checkpoints/nanoVLA-500M")

KEYS_TO_MODIFY_MAPPING = {
    # Vision Encoder: Non-block specific
    "vision_encoder.patch_embedding.conv.weight": "model.vision_model.embeddings.patch_embedding.weight",
    "vision_encoder.patch_embedding.conv.bias": "model.vision_model.embeddings.patch_embedding.bias",
    # HF stores position_embedding typically as '...position_embedding.weight'
    "vision_encoder.patch_embedding.position_embedding": "model.vision_model.embeddings.position_embedding.weight",
    "vision_encoder.layer_norm.weight": "model.vision_model.post_layernorm.weight",
    "vision_encoder.layer_norm.bias": "model.vision_model.post_layernorm.bias",

    # Vision Encoder Blocks (regex for block index and .weight/.bias suffix)
    # QKV weights (qkv_proj) are handled by a special splitting logic in convert_state_dict_to_hf
    r"vision_encoder\.blocks\.(\d+)\.ln1\.(weight|bias)": r"model.vision_model.encoder.layers.\1.layer_norm1.\2",
    r"vision_encoder\.blocks\.(\d+)\.attn\.out_proj\.(weight|bias)": r"model.vision_model.encoder.layers.\1.self_attn.out_proj.\2",
    r"vision_encoder\.blocks\.(\d+)\.ln2\.(weight|bias)": r"model.vision_model.encoder.layers.\1.layer_norm2.\2",
    r"vision_encoder\.blocks\.(\d+)\.mlp\.fc1\.(weight|bias)": r"model.vision_model.encoder.layers.\1.mlp.fc1.\2",
    r"vision_encoder\.blocks\.(\d+)\.mlp\.fc2\.(weight|bias)": r"model.vision_model.encoder.layers.\1.mlp.fc2.\2",

    # Language Model: Non-block specific
    "decoder.token_embedding.weight": "model.text_model.embed_tokens.weight",
    "decoder.norm.weight": "model.text_model.norm.weight",
    "decoder.head.weight": "lm_head.weight", # Idefics3 uses lm_head for the final output projection

    # Language Model Blocks (regex for block index and .weight/.bias suffix)
    r"decoder\.blocks\.(\d+)\.norm1\.(weight|bias)": r"model.text_model.layers.\1.input_layernorm.\2",
    r"decoder\.blocks\.(\d+)\.attn\.q_proj\.(weight|bias)": r"model.text_model.layers.\1.self_attn.q_proj.\2",
    r"decoder\.blocks\.(\d+)\.attn\.k_proj\.(weight|bias)": r"model.text_model.layers.\1.self_attn.k_proj.\2",
    r"decoder\.blocks\.(\d+)\.attn\.v_proj\.(weight|bias)": r"model.text_model.layers.\1.self_attn.v_proj.\2",
    r"decoder\.blocks\.(\d+)\.attn\.out_proj\.(weight|bias)": r"model.text_model.layers.\1.self_attn.o_proj.\2",
    r"decoder\.blocks\.(\d+)\.norm2\.(weight|bias)": r"model.text_model.layers.\1.post_attention_layernorm.\2",
    r"decoder\.blocks\.(\d+)\.mlp\.gate_proj\.(weight|bias)": r"model.text_model.layers.\1.mlp.gate_proj.\2",
    r"decoder\.blocks\.(\d+)\.mlp\.up_proj\.(weight|bias)": r"model.text_model.layers.\1.mlp.up_proj.\2",
    r"decoder\.blocks\.(\d+)\.mlp\.down_proj\.(weight|bias)": r"model.text_model.layers.\1.mlp.down_proj.\2",
    
    # Modality Projector
    # ModalityProjector in nanoVLM has self.proj = nn.Linear(..., bias=False)
    # So, only MP.proj.weight is expected.
    "MP.proj.weight": "model.connector.modality_projection.proj.weight",
}


WEIGHTS_TO_MERGE_MAPPING = (
)

WEIGHTS_TO_DROP = (
)

def convert_state_dict_to_hf(original_state_dict, vit_hidden_dim, keys_to_modify_mapping_dict, weights_to_drop_list):
    new_state_dict = {}
    for old_key, weight in original_state_dict.items():
        # Skip specified weights or rotary embedding inverse frequencies
        if old_key.endswith(".inv_freq") or any(w_to_drop in old_key for w_to_drop in weights_to_drop_list):
            continue
        current_key = old_key
        # 1. Special handling for Vision Transformer QKV weights: split combined qkv_proj
        # This rule takes precedence over general mappings for these specific keys.
        if "vision_encoder.blocks" in current_key and "attn.qkv_proj" in current_key:
            block_num_match = re.search(r"vision_encoder\.blocks\.(\d+)\.attn\.qkv_proj", current_key)
            if block_num_match:
                block_num = block_num_match.group(1)
                hf_prefix = f"model.vision_model.encoder.layers.{block_num}.self_attn."
                if current_key.endswith(".weight"):
                    # Weights are usually [3 * hidden_dim, hidden_dim]
                    # We need to split along dim 0 into three [hidden_dim, hidden_dim]
                    q_w, k_w, v_w = torch.split(weight, vit_hidden_dim, dim=0)
                    new_state_dict[hf_prefix + "q_proj.weight"] = q_w
                    new_state_dict[hf_prefix + "k_proj.weight"] = k_w
                    new_state_dict[hf_prefix + "v_proj.weight"] = v_w
                elif current_key.endswith(".bias"):
                    # Biases are usually [3 * hidden_dim]
                    # Split along dim 0 into three [hidden_dim]
                    q_b, k_b, v_b = torch.split(weight, vit_hidden_dim, dim=0)
                    new_state_dict[hf_prefix + "q_proj.bias"] = q_b
                    new_state_dict[hf_prefix + "k_proj.bias"] = k_b
                    new_state_dict[hf_prefix + "v_proj.bias"] = v_b
                continue # This key is fully processed, move to the next one in original_state_dict
        # 2. Apply general renaming rules from keys_to_modify_mapping_dict
        # The order of rules in keys_to_modify_mapping_dict can matter if patterns overlap.
        # This loop applies the *first* matching rule (regex or exact string).
        key_was_transformed = False
        for map_pattern, map_replacement in keys_to_modify_mapping_dict.items():
            # Check if map_pattern is likely a regex (contains regex special characters)
            # This heuristic can be refined if layer names themselves contain these chars.
            is_regex_pattern = any(c in map_pattern for c in r".*+?^$[]{}()|\1\2") 
            if is_regex_pattern:
                # If it's a regex, use re.sub.
                # re.sub returns the original string if no substitution is made.
                candidate_key, num_subs = re.subn(map_pattern, map_replacement, current_key)
                if num_subs > 0: # If a substitution actually happened
                    current_key = candidate_key
                    key_was_transformed = True
                    break # Rule applied, stop searching for this old_key
            else:
                # If not a regex, treat as a direct string match.
                if map_pattern == current_key:
                    current_key = map_replacement
                    key_was_transformed = True
                    break # Rule applied, stop searching for this old_key
        # After attempting to map, handle special cases for the (potentially transformed) key
        # Special handling for vision_encoder's position_embedding shape
        if current_key == "model.vision_model.embeddings.position_embedding.weight":
            # nanoVLM stores it as [1, num_patches (+1 if cls_token), embed_dim]
            # HF expects [num_patches (+1 if cls_token), embed_dim]
            if weight.ndim == 3 and weight.shape[0] == 1:
                weight = weight.squeeze(0)
        new_state_dict[current_key] = weight
    return new_state_dict


def merge_weights(state_dict, new_state_dict):
    old_weight_names = set(state_dict.keys())
    # Merge the weights
    for weights_to_merge, new_weight_name in WEIGHTS_TO_MERGE_MAPPING:
        for weight_to_merge in weights_to_merge:
            print(weight_to_merge)
            assert weight_to_merge in state_dict, f"Weight {weight_to_merge} is missing in the state dict"
            weight = state_dict.pop(weight_to_merge)
            if new_weight_name not in new_state_dict:
                new_state_dict[new_weight_name] = [weight]
            else:
                new_state_dict[new_weight_name].append(weight)
            old_weight_names.remove(weight_to_merge)
        new_state_dict[new_weight_name] = torch.cat(new_state_dict[new_weight_name], dim=0)
    # Remove the weights that were merged
    for weights_to_merge, new_weight_name in WEIGHTS_TO_MERGE_MAPPING:
        for weight in weights_to_merge:
            if weight in new_state_dict and weight != new_weight_name:
                new_state_dict.pop(weight)
    return new_state_dict

from dataclasses import asdict
from transformers import AutoConfig, Idefics3Config
def get_transformers_config(config):
    config_json = asdict(config)
    image_token_id = config_json.pop("image_token_id", config_json["lm_vocab_size"] + 1)
    config_json["lm_vocab_size"] = config_json.pop("lm_vocab_size") + config_json.pop("additional_vocab_size", 128)
    use_cache = config_json.pop("use_cache", True)
    tie_word_embeddings = config_json.pop("lm_tie_weights", True)
    scale_factor = config_json.pop("mp_pixel_shuffle_factor", 2)
    vocab_size = config_json.pop("lm_vocab_size", 100000)
    # Remove "freeze" params from the config
    text_config = AutoConfig.from_pretrained(config_json["lm_model_type"])
    text_config.vocab_size += 128 # Add 128 for the new special tokens
    vision_config = AutoConfig.from_pretrained(config_json['vit_model_type']).vision_config.to_dict()
    idefics_config = Idefics3Config(
        text_config=text_config,
        vision_config=vision_config,
        pad_token_id=text_config.pad_token_id,
        use_cache=use_cache,
        image_token_id=image_token_id,
        tie_word_embeddings=tie_word_embeddings,
        scale_factor=scale_factor,
        vocab_size=vocab_size,
    )
    return idefics_config


image_processor = Idefics3ImageProcessor()
processor = Idefics3Processor(
    image_processor=image_processor,
    tokenizer=tokenizer,
)
state_dict = nano_model.state_dict()
new_state_dict = convert_state_dict_to_hf(state_dict, config.vit_hidden_dim, KEYS_TO_MODIFY_MAPPING, WEIGHTS_TO_DROP)

new_state_dict = merge_weights(state_dict, new_state_dict)
del state_dict

transformers_config = get_transformers_config(config)
# print(transformers_config)

# config.text_config.pad_token_id = 2
# config.image_token_id = 49190
# config.text_config.vocab_size = 49280
# config.vision_config.tie_word_embeddings = False
# config.vision_config.num_attention_heads = 12
transformers_config.vision_config.max_image_size = {'longest_edge': transformers_config.vision_config.image_size}
transformers_config.vision_config.size = {"longest_edge": transformers_config.vision_config.image_size}

processor.image_seq_len = ((transformers_config.vision_config.image_size / transformers_config.vision_config.patch_size)  / transformers_config.scale_factor) ** 2
# processor.tokenizer.vocab_size = transformers_config.vocab_size

idefics_model = Idefics3ForConditionalGeneration(transformers_config)
idefics_model = idefics_model.to(torch.bfloat16)

# Resize embedding and lm_head weights
for key in ["model.text_model.embed_tokens.weight", "lm_head.weight"]:
    if key in new_state_dict:
        original_weight = new_state_dict[key]
        original_vocab_size, hidden_dim = original_weight.shape
        # Expected vocab size is from the model config
        if key == "model.text_model.embed_tokens.weight":
            expected_vocab_size = idefics_model.model.text_model.embed_tokens.weight.shape[0]
        elif key == "lm_head.weight":
            expected_vocab_size = idefics_model.lm_head.weight.shape[0]
        if original_vocab_size < expected_vocab_size:
            num_new_tokens = expected_vocab_size - original_vocab_size
            # Create new tensor with the target shape
            new_weight = torch.zeros(expected_vocab_size, hidden_dim, dtype=original_weight.dtype, device=original_weight.device)
            # Copy existing weights
            new_weight[:original_vocab_size, :] = original_weight
            # Initialize new token embeddings (e.g., with small random values)
            torch.nn.init.normal_(new_weight[original_vocab_size:, :], mean=0.0, std=0.02) # Common initialization
            new_state_dict[key] = new_weight
            print(f"Resized {key} from {original_weight.shape} to {new_weight.shape}")

idefics_model.load_state_dict(new_state_dict, strict=True)#, assign=True)

output_hub_path = "HuggingFaceTB/nanoVLA-500M-256"

# model.save_pretrained(output_hub_path)
# processor.save_pretrained(output_hub_path)

idefics_model.push_to_hub(output_hub_path, private=True, use_auth_token=True)
processor.push_to_hub(output_hub_path, private=True)
