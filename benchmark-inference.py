import torch
from PIL import Image

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor

from torch.utils import benchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16
use_torch_compile = False

print(f"Using device: {device} with dtype: {dtype}")

def generate_tokens(tokens, image):
    
    gen = model.generate(tokens, image, max_new_tokens=1000)


if __name__ == "__main__":
    model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-222M").to(device, dtype)
    model.eval()

    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)

    text = "What is this?"
    template = f"Question: {text} Answer:"
    encoded_batch = tokenizer.batch_encode_plus([template], return_tensors="pt")
    tokens = encoded_batch["input_ids"].to(device)

    image_path = "assets/image.png"
    image = Image.open(image_path)
    image = image_processor(image)
    image = image.unsqueeze(0).to(device, dtype)

    if use_torch_compile:
        model.decoder = torch.compile(model.decoder, mode="reduce-overhead", fullgraph=True)

    # Warmup
    for i in range(3):
        _ = model.generate(tokens, image, max_new_tokens=128)

    time = benchmark.Timer(
        stmt="generate_tokens(tokens, image)",
        setup="from __main__ import generate_tokens",
        globals={"tokens": tokens, "image": image},
        num_threads=torch.get_num_threads(),
    )

    print(time.timeit(10))
