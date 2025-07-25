import argparse
import torch
from PIL import Image

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from an image with nanoVLM")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a local checkpoint (directory or safetensors/pth). If omitted, we pull from HF."
    )
    parser.add_argument(
        "--hf_model", type=str, default="lusxvr/nanoVLM-450M",
        help="HuggingFace repo ID to download from incase --checkpoint isnt set."
    )
    parser.add_argument("--image", type=str, default="assets/image.png",
                        help="Path to input image")
    parser.add_argument("--prompt", type=str, default="What is this?",
                        help="Text prompt to feed the model")
    parser.add_argument("--generations", type=int, default=5,
                        help="Num. of outputs to generate")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Maximum number of tokens per output")
    return parser.parse_args()


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
    # FIXED: Use vit_img_size and splitted_image_size for MPS compatibility
    # Original: get_image_processor(model.cfg.max_img_size, model.cfg.vit_img_size)
    image_processor = get_image_processor(model.cfg.vit_img_size, model.cfg.splitted_image_size)

    img = Image.open(args.image).convert("RGB")
    # FIXED: Simplified image processing for MPS compatibility
    # Original: processed_image, splittedimage_count = image_processor(img)
    img_t = image_processor(img)[0][0].unsqueeze(0).to(device)
    vit_patch_size = splittedimage_count[0] * splittedimage_count[1]

    messages = [{"role": "user", "content": tokenizer.image_token * model.cfg.mp_image_token_length * vit_patch_size + args.prompt}]
    encoded_prompt = tokenizer.apply_chat_template([messages], tokenize=True, add_generation_prompt=True)
    tokens = torch.tensor(encoded_prompt).to(device)
    img_t = processed_image.to(device)

    print("\nInput:\n ", args.prompt, "\n\nOutputs:")
    for i in range(args.generations):
        gen = model.generate(tokens, img_t, max_new_tokens=args.max_new_tokens)
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        print(f"  >> Generation {i+1}: {out}")


if __name__ == "__main__":
    main()
