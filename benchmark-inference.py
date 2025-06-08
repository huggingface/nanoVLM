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
use_torch_compile = True
use_kv_cache = True
kv_cache_implementation = "static"


if use_torch_compile:
    if not use_kv_cache:
        raise ValueError("KV cache must be enabled when using torch compile")
    if kv_cache_implementation != "static":
        raise ValueError("KV cache implementation must be static when using torch compile")

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True

torch._logging.set_logs(graph_breaks=True, recompiles=True)


print(f"Using device: {device} with dtype: {dtype}")


def generate_tokens(tokens, image):
    gen = model.generate(tokens, image, max_new_tokens=1000, use_kv_cache=use_kv_cache, kv_cache_implementation=kv_cache_implementation)


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
        model.decoder.forward = torch.compile(model.decoder.forward, mode="max-autotune", fullgraph=True)

    # Warmup
    for i in range(3):
        print(f"Warmup {i+1}")
        _ = model.generate(tokens, image, max_new_tokens=1000, use_kv_cache=True, kv_cache_implementation=kv_cache_implementation)

    time = benchmark.Timer(
        stmt="generate_tokens(tokens, image)",
        setup="from __main__ import generate_tokens",
        globals={"tokens": tokens, "image": image},
        num_threads=torch.get_num_threads(),
    )

    print(time.timeit(10))
