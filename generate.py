import argparse

import torch
from PIL import Image

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from data.processors import get_image_processor, get_image_string, get_tokenizer
from models.vision_language_model import VisionLanguageModel
from smolagents import ChatMessage, MessageRole


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from an image with nanoVLM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a local checkpoint (directory or safetensors/pth). If omitted, we pull from HF.",
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        default="lusxvr/nanoVLM-460M",
        help="HuggingFace repo ID to download from incase --checkpoint isnt set.",
    )
    parser.add_argument("--image", type=str, default="assets/image.png", help="Path to input image")
    parser.add_argument("--prompt", type=str, default="What is this?", help="Text prompt to feed the model")
    parser.add_argument("--generations", type=int, default=5, help="Num. of outputs to generate")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Maximum number of tokens per output")
    return parser.parse_args()


class NanoVLMModel:
    def __init__(self, model_id: str = "lusxvr/nanoVLM-460M", to_device: str = "cuda"):
        self.model_id = model_id
        self.device = to_device
        self.model = VisionLanguageModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

        self.tokenizer = get_tokenizer(self.model.cfg.lm_tokenizer, self.model.cfg.vlm_extra_tokens)
        self.image_processor = get_image_processor(
            self.model.cfg.max_img_size, self.model.cfg.vit_img_size, self.model.cfg.resize_to_max_side_len
        )

    def generate(self, messages: list[dict], **kwargs):
        if len(messages) > 1:
            raise ValueError("NanoVLMModel only supports one message")
        new_messages_str = []
        for msg in messages[0]["content"]:
            if msg["type"] == "image":
                img = msg["image"]
            elif msg["type"] == "text":
                new_messages_str.append(msg["text"])
        new_messages_str = "\n".join(new_messages_str)

        img = img.convert("RGB")
        processed_image, splitted_image_ratio = self.image_processor(img)
        if (
            not hasattr(self.tokenizer, "global_image_token")
            and splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1
        ):
            # If the tokenizer doesn't have a global image token, but the processor generated it, remove it
            processed_image = processed_image[1:]

        image_string = get_image_string(self.tokenizer, [splitted_image_ratio], self.model.cfg.mp_image_token_length)
        messages = [{"role": "user", "content": image_string + new_messages_str}]
        encoded_prompt = self.tokenizer.apply_chat_template([messages], tokenize=True, add_generation_prompt=True)
        tokens = torch.tensor(encoded_prompt).to(self.device)
        img_t = processed_image.to(self.device)
        kwargs["max_new_tokens"] = 100
        gen = self.model.generate(tokens, img_t, **kwargs)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0],
        )


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    source = args.checkpoint if args.checkpoint else args.hf_model
    print(f"Loading weights from: {source}")
    model = VisionLanguageModel.from_pretrained(source).to(device)
    model.eval()

    tokenizer = get_tokenizer(model.cfg.lm_tokenizer, model.cfg.vlm_extra_tokens)
    image_processor = get_image_processor(model.cfg.max_img_size, model.cfg.vit_img_size)

    img = Image.open(args.image).convert("RGB")
    processed_image, splitted_image_ratio = image_processor(img)
    if (
        not hasattr(tokenizer, "global_image_token")
        and splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1
    ):
        # If the tokenizer doesn't have a global image token, but the processor generated it, remove it
        processed_image = processed_image[1:]

    image_string = get_image_string(tokenizer, [splitted_image_ratio], model.cfg.mp_image_token_length)

    messages = [{"role": "user", "content": image_string + args.prompt}]
    encoded_prompt = tokenizer.apply_chat_template([messages], tokenize=True, add_generation_prompt=True)
    tokens = torch.tensor(encoded_prompt).to(device)
    img_t = processed_image.to(device)

    print("\nInput:\n ", args.prompt, "\n\nOutputs:")
    for i in range(args.generations):
        gen = model.generate(tokens, img_t, max_new_tokens=args.max_new_tokens)
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        print(f"  >> Generation {i + 1}: {out}")


if __name__ == "__main__":
    main()
