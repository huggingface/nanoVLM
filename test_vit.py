from models.config import VLMConfig
from models.vision_transformer import ViT
from transformers import AutoProcessor, SiglipVisionModel
from transformers.image_utils import load_image
import torch

def test_equivalnce():
    vlm_config = VLMConfig()
    vit_model = ViT.from_pretrained(vlm_config)
    vit_model.eval()

    model_id = vlm_config.vit_model_type
    processor = AutoProcessor.from_pretrained(vlm_config.vit_model_type)
    vit_model_hf = SiglipVisionModel.from_pretrained(model_id)
    vit_model_hf.eval()

    image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        outputs = vit_model(pixel_values)
        hf_outputs = vit_model_hf(pixel_values).last_hidden_state

    assert hf_outputs.shape == outputs.shape, (
        f"[{model_id}] Shape mismatch: HF={hf_outputs.shape}, Custom={outputs.shape}"
    )

    torch.testing.assert_close(
        hf_outputs,
        outputs,
        rtol=1e-4,
        atol=1e-5,
        msg=f"[{model_id}] Model outputs are not numerically close",
    )

if __name__ == "__main__":
    test_equivalnce()