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
from data.processors import get_tokenizer, get_image_processor, get_image_string
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
        resize_to_max_side_len = False
        if hasattr(self.model.cfg, "resize_to_max_side_len"):
            resize_to_max_side_len = self.model.cfg.resize_to_max_side_len
        print(f"Resize to max side len: {resize_to_max_side_len}")
        self.image_processor = get_image_processor(self.model.cfg.max_img_size, self.model.cfg.vit_img_size, resize_to_max_side_len)
            
    def _prepare_visual_input(self, visual_list: List[Image.Image]) -> Optional[torch.Tensor]:
        """Convert visual inputs to model format."""
        if not visual_list or visual_list[0] is None: # Still check if the list is empty or contains None
            return None, None
            
        images = []
        splitted_image_ratios = []
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
            processed_images, splitted_image_ratio = self.image_processor(image)
            if not hasattr(self.tokenizer, "global_image_token") and splitted_image_ratio[0]*splitted_image_ratio[1] == len(processed_images) - 1:
                # If the tokenizer doesn't have a global image token, but the processor generated it, remove it
                processed_images = processed_images[1:]

            images.append(processed_images)
            splitted_image_ratios.append(splitted_image_ratio)
        
        if images:
            return images, splitted_image_ratios
        return None, None
        
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for nanoVLM")

    def flatten(self, input):
        new_list = []
        for sublist in input:
            if sublist is None:
                new_list.append(None)
            else:
                for i in sublist:
                    new_list.append(i)
        return new_list
    
    def get_benchmark_formatting(self, task_name: str) -> dict:
        """Get benchmark-specific formatting rules."""
        benchmark_formats = {
            ("ai2d", "mmstar", "seedbench", "scienceqa"): { #   
                "text_replacements": {
                    "\nOptions:": "\nChoices:",
                    "\nA. ": "\nChoices:\nA. ",
                    "Please select the correct answer from the options above.": "Answer with the letter.",
                    "Answer with the option's letter from the given choices directly": "Answer with the letter directly",
                },
                "assistant_prefix": "Answer:",
                "user_prefix": "",
                "user_suffix": ""
            },
            ("docvqa_val", "docvqa_test"): {
                "text_replacements": {},
                "assistant_prefix": "",
                "user_prefix": "Give a short and terse answer to the following question. "
                                + "Do not paraphrase or reformat the text you see in the image. Do not include any full stops. "
                                + "Just give the answer without additional explanation. Question: ",
                "user_suffix": ""
            },
            "chartvqa": {
                "text_replacements": {},
                "assistant_prefix": "",
                "user_prefix": "For the question below, follow the following instructions:\n"
                                + "-The answer should contain as few words as possible.\n"
                                + "-Don't paraphrase or reformat the text you see in the image.\n"
                                + "-Answer a binary question with Yes or No.\n"
                                + "-When asked to give a numerical value, provide a number like 2 instead of Two.\n"
                                + "-If the final answer has two or more items, provide it in the list format like [1, 2].\n"
                                + "-When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4.\n"
                                + "-When asked to give a percentage, give out the whole value like 17 instead of decimal like 0.17%.\n"
                                + "-Don't include any units in the answer.\n"
                                + "-Do not include any full stops at the end of the answer.\n"
                                + "-Try to include the full label from the graph when asked about an entity.\n"
                                + "Question: ",
                "user_suffix": ""
            },
            ("textvqa_val", "textvqa_test"): {
                "text_replacements": {},
                "assistant_prefix": "",
                "user_prefix": "Answer the following question about the image using as few words as possible. "
                                + "Follow these additional instructions:\n"
                                + "-Always answer a binary question with Yes or No.\n"
                                + "-When asked what time it is, reply with the time seen in the image.\n"
                                + "-Do not put any full stops at the end of the answer.\n"
                                + "-Do not put quotation marks around the answer.\n"
                                + "-An answer with one or two words is favorable.\n"
                                + "-Do not apply common sense knowledge. The answer can be found in the image.\n"
                                + "Question: ",
                "user_suffix": ""
            },
            ("mmmu_val", "mmmu_test"): {
                "text_replacements": {
                    "Question:": "",
                    "Answer with the option's letter from the given choices directly.": "Answer with the letter directly.",
                    "\nA. ": "\nChoices:\nA. "
                },
                "assistant_prefix": "Answer:",
                "user_prefix": "",
                "user_suffix": ""
            },
            ("infovqa_val", "mme", "ocrbench"): {
                "text_replacements": {},
                "assistant_prefix": "",
                "user_prefix": "",
                "user_suffix": "\nGive a very brief answer."
            }
        }
        
        # Check individual task names first
        if task_name in benchmark_formats:
            return benchmark_formats[task_name]
        
        # Check if task is in any list/tuple keys
        for key, formatting in benchmark_formats.items():
            if isinstance(key, (list, tuple)) and task_name in key:
                return formatting
        
        # Default formatting
        return {"text_replacements": {}, "assistant_prefix": "", "user_prefix": "", "user_suffix": ""}
    
    def apply_benchmark_formatting(self, context_str: str, prompt: str, task_name: str) -> tuple[str, str]:
        """Apply benchmark-specific formatting to context and prompt."""
        formatting = self.get_benchmark_formatting(task_name)
        
        # Add user prefix to context
        if formatting["user_prefix"]:
            context_str = formatting["user_prefix"] + context_str
        
        # Apply text replacements to context
        for old_text, new_text in formatting["text_replacements"].items():
            context_str = context_str.replace(old_text, new_text)
        
        # Add user suffix to context
        if formatting["user_suffix"]:
            context_str = context_str + formatting["user_suffix"]
        
        # Add assistant prefix to prompt
        if formatting["assistant_prefix"]:
            prompt = prompt + formatting["assistant_prefix"]
        
        return context_str, prompt
    
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
            try:
                contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
                visuals = [dtv(self.task_dict[t][s][i]) for dtv, i, t, s in zip(doc_to_visual, doc_id, task, split)]
                images, splitted_image_ratio = self._prepare_visual_input(self.flatten(visuals))
            except Exception as e:
                print(f"Error preparing visual input: {e}")
                if len(contexts) > 0:
                    pbar.update(len(contexts))
                    generated_texts = [""] * len(contexts)
                    res.extend(generated_texts)
                continue

            messages = []
            splitted_image_idx = 0
            for i in range(len(contexts)):
                current_context_str = contexts[i]
                
                # Apply benchmark-specific text replacements
                current_context_str, _ = self.apply_benchmark_formatting(current_context_str, "", task[i])
                
                if visuals[i] is None:
                    image_count = 0
                else:
                    image_count = len(visuals[i])
                image_string = ""
                for _ in range(image_count):
                    image_string += get_image_string(self.tokenizer, [splitted_image_ratio[splitted_image_idx]], self.model.cfg.mp_image_token_length)
                    splitted_image_idx += 1

                prompt_content = image_string + current_context_str
                
                # Format text_data as a list of message dictionaries
                messages_for_item = [{"role": "user", "content": prompt_content}]
                messages.append(messages_for_item)
                
                # # Process images; _prepare_visual_input returns a stacked tensor or None
                # processed_images_tensor = self._prepare_visual_input(current_visuals_list) if current_visuals_list else None
                # images.append(processed_images_tensor)
                
            prompts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            pr = False
            if pr:
                print(task[0])
                print("Original Prompt")
                print(prompts[0])

            # Apply benchmark-specific assistant prefixes
            for i in range(len(prompts)):
                _, prompts[i] = self.apply_benchmark_formatting("", prompts[i], task[i])

            if pr:
                print("Formatted Prompt")
                print(prompts[0])

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                padding_side="left",
                truncation=True,
                max_length=self.max_length
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            # images = images.to(self.device)

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
            if pr:
                print(generated_texts[0])
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