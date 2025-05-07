import torch
from PIL import Image

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model from the Hugging Face Hub or a local directory
model = VisionLanguageModel.from_pretrained("ariG23498/nanoVLM-demo").to(device)

model.eval()

tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
image_processor = get_image_processor(model.cfg.vit_img_size)

text = "What is this?"
template = f"Question: {text} Answer:"
encoded_batch = tokenizer.batch_encode_plus([template], return_tensors="pt")
tokens = encoded_batch['input_ids'].to(device)

image_path = 'assets/image.png'
image = Image.open(image_path)
image = image_processor(image)
image = image.unsqueeze(0).to(device)

print("Input: ")
print(f'{text}')
print("Output:")
num_generations = 5
for i in range(num_generations):
    gen = model.generate(tokens, image, max_new_tokens=20)
    print(f"Generation {i+1}: {tokenizer.batch_decode(gen, skip_special_tokens=True)[0]}")