from models.config import VLMConfig
from models.language_model import LanguageModel
from transformers import AutoTokenizer, LlamaForCausalLM
import torch

def test_equivalnce():
    vlm_config = VLMConfig(lm_use_tokens=True)
    lm_model = LanguageModel.from_pretrained(vlm_config)
    lm_model.eval()

    model_id = vlm_config.lm_model_type
    tokenizer_id = vlm_config.lm_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    lm_model_hf = LlamaForCausalLM.from_pretrained(model_id)
    lm_model_hf.eval()

    prompt = "Hello there!"
    inputs = tokenizer(text=prompt, return_tensors="pt")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = lm_model(input_ids, attention_mask)
        hf_outputs = lm_model_hf(input_ids, attention_mask).logits

    assert hf_outputs.shape == outputs.shape, (
        f"[{model_id}] Shape mismatch: HF={hf_outputs.shape}, Custom={outputs.shape}"
    )

    torch.testing.assert_close(
        hf_outputs,
        outputs,
        rtol=1e-5,
        atol=1e-5,
        msg=f"[{model_id}] Model outputs are not numerically close",
    )

if __name__ == "__main__":
    test_equivalnce()