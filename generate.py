import argparse
import torch
from PIL import Image
from huggingface_hub import hf_hub_download


from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig
from data.processors import get_tokenizer, get_image_processor

# ++++++++++++++++++++++++++++++++++++++++++++++++++++
# generate.py
#
# • Added argparse flags:
#     --checkpoint     Path to a local .pth model (falls back to HF hub if omitted)
#     --image          Path to input image file (default: assets/image.png)
#     --prompt         Text prompt for the model (default: "What is this?")
#     --generations    Number of outputs to generate (default: 5)
#     --max_new_tokens Maximum tokens per generation (default: 20)
#
# Wraps logic in main(), parses args, loads either local checkpoint or hub weights
# then runs the vision+language pipeline to produce the requested generations.
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def parse_args():
    parser = argparse.ArgumentParser(
        description="generate text from an image with nanoVLM"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a local .pth checkpoint (if provided). If omitted, downloads from HuggingFace Hub."
    )
    parser.add_argument(
        "--image", type=str, default="assets/image.png",
        help="Path to input image"
    )
    parser.add_argument(
        "--prompt", type=str, default="What is this?",
        help="Text prompt to feed the model"
    )
    parser.add_argument(
        "--generations", type=int, default=5,
        help="Number of outputs to generate"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=20,
        help="Maximum number of tokens to generate"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = VLMConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = hf_hub_download(
            repo_id="lusxvr/nanoVLM-222M",
            filename="nanoVLM-222M.pth"
        )

    model = VisionLanguageModel(cfg).to(device)
    model.load_checkpoint(checkpoint_path)
    model.eval()

    tokenizer = get_tokenizer(cfg.lm_tokenizer)
    image_processor = get_image_processor(cfg.vit_img_size)

    text = args.prompt
    template = f"Question: {text} Answer:"
    encoded = tokenizer.batch_encode_plus([template], return_tensors="pt")
    tokens = encoded["input_ids"].to(device)

    image = Image.open(args.image).convert("RGB")
    image_tensor = image_processor(image).unsqueeze(0).to(device)

    print("\nInput:")
    print(text)
    print("\nOutput:")
    for i in range(args.generations):
        gen_ids = model.generate(tokens,
                                 image_tensor,
                                 max_new_tokens=args.max_new_tokens)
        output = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        print(f"Generation {i+1}: {output}")


if __name__ == "__main__":
    main()
