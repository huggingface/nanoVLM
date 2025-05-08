import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from accelerate import Accelerator

from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig
from data.processors import get_tokenizer, get_image_processor

torch.manual_seed(0)

# Initialize accelerator
accelerator = Accelerator(mixed_precision="bf16" if torch.cuda.is_available() else "no")

cfg = VLMConfig()
# Device is now handled by accelerator
print(f"Using device: {accelerator.device}")

# Change to your own model path after training
path_to_hf_file = hf_hub_download(repo_id="lusxvr/nanoVLM-222M", filename="nanoVLM-222M.pth")
model = VisionLanguageModel(cfg)
model.load_checkpoint(path_to_hf_file)
# Prepare model with accelerator
model = accelerator.prepare(model)
model.eval()

tokenizer = get_tokenizer(cfg.lm_tokenizer)
image_processor = get_image_processor(cfg.vit_img_size)

text = "What is this?"
template = f"Question: {text} Answer:"
encoded_batch = tokenizer.batch_encode_plus([template], return_tensors="pt")
tokens = encoded_batch['input_ids']

image_path = 'assets/image.png'
image = Image.open(image_path)
image = image_processor(image)
image = image.unsqueeze(0)

tokens, image = accelerator.prepare(tokens, image)

print("Input: ")
print(f'{text}')
print("Output:")
num_generations = 5
for i in range(num_generations):
    gen = model.generate(tokens, image, max_new_tokens=20)
    print(f"Generation {i+1}: {tokenizer.batch_decode(gen, skip_special_tokens=True)[0]}")