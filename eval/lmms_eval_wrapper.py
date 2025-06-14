"""
LMMS-Eval wrapper for nanoVLM model.
This allows using lmms-eval for intermediate evaluation during training.
"""

import torch
from typing import List, Tuple, Optional, Union
from PIL import Image
import numpy as np
import torch.distributed as dist

from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor
from data.collators import VQACollator


class NanoVLMWrapper(lmms):
    """Wrapper to make nanoVLM compatible with lmms-eval framework."""
    
    def __init__(
        self,
        model: str | VisionLanguageModel = "lusxvr/nanoVLM-450M",
        device: str = "cuda",
        batch_size: int = 32,
        **kwargs
    ):
        super().__init__()
        if isinstance(model, str):
            self.model = VisionLanguageModel.from_pretrained(model).to(device)
        else:
            self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        
        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            # Fallback for non-distributed execution
            self._rank = 0
            self._world_size = 1
        
        # Get tokenizer and image processor from model config if not provided
        self.tokenizer = get_tokenizer(self.model.cfg.lm_tokenizer, self.model.cfg.vlm_extra_tokens, self.model.cfg.lm_chat_template)
        self.image_processor = get_image_processor(self.model.cfg.vit_img_size)
            
    def _prepare_visual_input(self, visual_list: List[Image.Image]) -> Optional[torch.Tensor]:
        """Convert visual inputs to model format."""
        if not visual_list or visual_list[0] is None: # Still check if the list is empty or contains None
            return None
            
        images = []
        for visual in visual_list:
            image = None
            if isinstance(visual, Image.Image):
                image = visual
            elif isinstance(visual, str): # Keep path loading for convenience
                image = Image.open(visual).convert("RGB")
            elif isinstance(visual, np.ndarray): # Keep numpy array loading for convenience
                image = Image.fromarray(visual)
            else:
                # If it's not an Image, a path string, or a numpy array, it's an error
                raise ValueError(f"Unsupported visual type: {type(visual)}. Expected PIL Image, path string, or numpy array.")
            
            # Process image
            processed = self.image_processor(image)
            images.append(processed)
        
        if images:
            return torch.stack(images).to(self.device)
        return None
        
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for nanoVLM")

    def flatten(self, input):
        new_list = []
        for sublist in input:
            for i in sublist:
                new_list.append(i)
        return new_list
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids, task, split in zip(doc_id, task, split)]
            images = self._prepare_visual_input(self.flatten(visuals))

            messages = []
            for i in range(len(contexts)):
                current_context_str = contexts[i]
                current_visuals_list = visuals[i] # List of PIL Images for this sample, or None

                num_images_for_item = 0
                if current_visuals_list: # Check if the list is not None and not empty
                    num_images_for_item = len(current_visuals_list)
                
                # Prepend image tokens based on the number of images for the current item
                image_tokens_prefix = self.tokenizer.image_token * num_images_for_item * self.model.cfg.mp_image_token_length
                prompt_content = image_tokens_prefix + current_context_str
                
                # Format text_data as a list of message dictionaries
                messages_for_item = [{"role": "user", "content": prompt_content}]
                messages.append(messages_for_item)
                
                # # Process images; _prepare_visual_input returns a stacked tensor or None
                # processed_images_tensor = self._prepare_visual_input(current_visuals_list) if current_visuals_list else None
                # images.append(processed_images_tensor)
                
            prompts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_length
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            images = images.to(self.device)

            # Extract generation parameters for the batch
            # We use the gen_kwargs from the first item in the chunk, assuming they are uniform for the batch.
            # lmms-eval groups requests by gen_kwargs, so this assumption should hold.
            current_gen_kwargs = all_gen_kwargs[0] if all_gen_kwargs else {}
            max_new_tokens = current_gen_kwargs.get("max_new_tokens", 50)
            temperature = current_gen_kwargs.get("temperature", 0.0) # Default to greedy
            top_p = current_gen_kwargs.get("top_p", 1.0)
            # Check if greedy generation is explicitly requested or implied by temperature 0
            greedy = current_gen_kwargs.get("do_sample", False) is False or temperature == 0.0
            # Pass None for temperature/top_p if greedy, as some HF models expect this
            gen_temperature = temperature if not greedy else None
            gen_top_p = top_p if not greedy else None
            
            # Generate
            generated_ids_batch = self.model.generate(
                input_ids,
                images,
                attention_mask,
                max_new_tokens=max_new_tokens,
                greedy=greedy,
                temperature=gen_temperature,
                top_p=gen_top_p,
            )

            # Decode generated sequences
            # generated_ids_batch from model.generate usually contains only the generated tokens (excluding prompt)
            generated_texts = self.tokenizer.batch_decode(
                generated_ids_batch,
                skip_special_tokens=True
            )
            res.extend(generated_texts)
            pbar.update(len(contexts))

        pbar.close()

        # print(res)
        # re_ords.get_original() will sort the results back to the original order of requests
        return re_ords.get_original(res)

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Multi Round Generation is not implemented for nanoVLM")
    
    @property
    def max_length(self):
        """Return the maximum sequence length."""
        return self.model.cfg.lm_max_position_embeddings 
    
    @property
    def batch_size_per_gpu(self):
        """Return the batch size."""
        return self.batch_size