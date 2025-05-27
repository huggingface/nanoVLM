import torch
import unittest
from models.language_model import LanguageModel
from types import SimpleNamespace

class TestLanguageModel(unittest.TestCase):
    def setUp(self):
        # Minimal config for testing
        self.cfg = SimpleNamespace(
            lm_hidden_dim=64,
            lm_inter_dim=128,
            lm_rms_eps=1e-5,
            lm_re_base=10000.0,
            lm_max_position_embeddings=1024,
            lm_attn_scaling=1.0,
            lm_vocab_size=100, # Small vocab for testing
            lm_n_heads=4,
            lm_n_kv_heads=2,
            lm_dropout=0.0,
            lm_n_blocks=2,
            lm_use_tokens=True,
            lm_tie_weights=True
        )
        self.model = LanguageModel(self.cfg)
        self.model.eval() # Set model to evaluation mode

    def test_kv_caching_consistency(self):
        # Input for the model
        batch_size = 16
        seq_len = 1000
        input_ids = torch.randint(0, self.cfg.lm_vocab_size, (batch_size, seq_len))

        # Forward pass without KV caching (prefill)
        output_no_cache, _ = self.model(input_ids, start_pos=0)

        # Forward pass with KV caching
        # 1. Prefill phase
        prefill_output, kv_cache_prefill = self.model(input_ids[:, :-1], start_pos=0)
        
        # 2. Decode phase (one token at a time)
        # We expect the output of the last token from prefill + decode to match the no_cache output
        # for that same token.
        
        # Get the last token's input_id for the decode step
        last_token_input = input_ids[:, -1].unsqueeze(-1) # Shape: [B, 1]
        
        # The start_pos for this token is seq_len - 1
        output_with_cache_last_token, _ = self.model(
            last_token_input, 
            kv_cache=kv_cache_prefill, 
            start_pos=seq_len - 1
        )

        # Compare the logits for the last token
        # output_no_cache is [B, seq_len, vocab_size]
        # output_with_cache_last_token is [B, 1, vocab_size]
        
        # We compare the last token's output from the no_cache run
        # with the single token output from the with_cache run.
        logits_no_cache_last_token = output_no_cache[:, -1, :]
        logits_with_cache_last_token = output_with_cache_last_token[:, 0, :]

        self.assertTrue(
            torch.allclose(logits_no_cache_last_token, logits_with_cache_last_token, atol=1e-5),
            "Outputs with and without KV caching do not match for the last token."
        )

        # Let's also test a multi-step decode to be more thorough
        # We'll compare the full sequence output if we decode token by token
        
        # Reset for a full token-by-token generation using KV cache
        current_input = input_ids[:, :1] # Start with the first token
        output_tokens_with_cache_list = []
        kv_cache_step = None

        for i in range(seq_len):
            if i > 0:
                current_input = input_ids[:, i:i+1] # Next token
            
            # The start_pos for the current token is simply i
            output_step, kv_cache_step = self.model(
                current_input,
                kv_cache=kv_cache_step,
                start_pos=i 
            )
            # output_step is [B, 1, vocab_size]
            output_tokens_with_cache_list.append(output_step)
        
        # Concatenate all single token outputs
        output_with_cache_full = torch.cat(output_tokens_with_cache_list, dim=1) # [B, seq_len, vocab_size]

        self.assertTrue(
            torch.allclose(output_no_cache, output_with_cache_full, atol=1e-5),
            "Full sequence outputs with and without KV caching do not match."
        )

if __name__ == '__main__':
    unittest.main() 