from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor
from huggingface_hub import hf_hub_download
from transformers.image_utils import load_image
import torch

def test_equivalence():
    # Create the model with random weights
    print("Create the model with random weights")
    config = VLMConfig()
    config.vlm_load_backbone_weights = False
    model = VisionLanguageModel(config)

    # Load the original `.pth` weights
    print("Load the original `.pth` weights")
    file_path = hf_hub_download(repo_id="lusxvr/nanoVLM-222M", filename="nanoVLM-222M.pth")
    model.load_state_dict(state_dict=torch.load(file_path))

    # Save the model's weights in safetensors
    print("Save the model's weights in safetensors")
    model.save_pretrained(save_directory="nanovlm-safe")

    # Load another model with the safetensors
    print("Create a new model with safetensor weights")
    model_safe = VisionLanguageModel.from_pretrained(repo_id_or_path="nanovlm-safe")

    # Prepare inputs
    print("Prepare the inputs")
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)

    text = "What is this?"
    template = f"Question: {text} Answer:"
    encoded_batch = tokenizer.batch_encode_plus([template], return_tensors="pt")
    tokens = encoded_batch['input_ids']

    image_path = 'assets/image.png'
    image = load_image(image_path)
    image = image_processor(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs =  model(tokens, image)[0]
        outputs_safe = model_safe(tokens, image)[0]
    
    assert outputs.shape == outputs_safe.shape, (
        f"Outputs Shape mismatch: torch={outputs.shape}, safetensors={outputs_safe.shape}"
    )

    print("Run equivalnces")
    torch.testing.assert_close(
        outputs,
        outputs_safe,
        rtol=1e-4,
        atol=1e-5,
        msg=f"model outputs are not numerically close",
    )

if __name__ == "__main__":
    test_equivalence()