import torch
import time
from tqdm import tqdm
from PIL import Image

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def generate_tokens(tokens, image):
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            return model.generate(tokens, image, max_new_tokens=100)

if __name__ == "__main__":
    model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-222M").to(device)
    model.eval()

    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)

    # 5) Prepare inputs
    text = "What is this?"
    template = f"Question: {text} Answer:"
    encoded_batch = tokenizer.batch_encode_plus([template], return_tensors="pt")
    tokens = encoded_batch["input_ids"].to(device)

    image_path = "assets/image.png"
    image = Image.open(image_path)
    image = image_processor(image).unsqueeze(0).to(device)

    num_runs = 10

    _ = generate_tokens(tokens, image)

    durations = []
    for _ in tqdm(range(num_runs), desc="Running inference"):
        t0 = time.perf_counter()
        _ = generate_tokens(tokens, image)
        durations.append(time.perf_counter() - t0)

    avg_time = sum(durations) / len(durations)
    print(f"\nAverage inference time over {num_runs} runs: {avg_time:.4f}s")
