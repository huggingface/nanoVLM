import torch; torch.manual_seed(0);
from PIL import Image

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor

torch.manual_seed(0)

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

def generate_tokens(tokens, image):
    gen = model.generate(tokens, image, max_new_tokens=100)
    return gen

if __name__ == "__main__":
    model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-222M").to(device)
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
    for i in range(5):
        gen = generate_tokens(tokens, image)
        print(f"Generation {i+1}: {tokenizer.batch_decode(gen, skip_special_tokens=True)[0]}")
