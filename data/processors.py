from transformers import AutoTokenizer
import torchvision.transforms as transforms

TOKENIZERS_CACHE = {}

def get_tokenizer(name, extra_special_tokens=None, chat_template=None):
    if name not in TOKENIZERS_CACHE:
        tokenizer_init_kwargs = {"use_fast": True}
        if extra_special_tokens is not None:
            tokenizer_init_kwargs["extra_special_tokens"] = extra_special_tokens
        if chat_template is not None:
            tokenizer_init_kwargs["chat_template"] = chat_template
        tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_init_kwargs,)
        tokenizer.pad_token = tokenizer.eos_token
        TOKENIZERS_CACHE[name] = tokenizer
    return TOKENIZERS_CACHE[name]

from transformers import Siglip2ImageProcessor

def get_image_processor(img_size):
    return Siglip2ImageProcessor(
        max_num_patches=1024,
    )

# def get_image_processor(img_size):
#     return transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor()
#     ])
