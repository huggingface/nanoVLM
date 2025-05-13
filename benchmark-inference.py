import os
import sys
import time
import torch
import torch.cuda
from PIL import Image
from typing import Dict, Any
from torch.utils import benchmark

# Dynamically add project root to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    from models.vision_language_model import VisionLanguageModel
    from data.processors import get_tokenizer, get_image_processor
except ImportError as e:
    print(f"Import Error: {e}")
    print("Possible solutions:")
    print("1. Ensure project structure is correct")
    print("2. Check PYTHONPATH")
    print("3. Verify import paths in models and data directories")
    sys.exit(1)

class VLMBenchmark:
    def __init__(self, model_name: str = "lusxvr/nanoVLM-222M"):
        """
        Initialize VLM Benchmark with device selection and model loading

        Args:
            model_name (str): Pretrained model identifier
        """
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Model and processor initialization
        try:
            self.model = VisionLanguageModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Model Loading Error: {e}")
            print("Possible solutions:")
            print("1. Check model name and availability")
            print("2. Verify model loading method")
            sys.exit(1)
        
        try:
            self.tokenizer = get_tokenizer(self.model.cfg.lm_tokenizer)
            self.image_processor = get_image_processor(self.model.cfg.vit_img_size)
        except Exception as e:
            print(f"Processor Initialization Error: {e}")
            print("Possible solutions:")
            print("1. Check tokenizer and image processor configurations")
            print("2. Verify get_tokenizer and get_image_processor implementations")
            sys.exit(1)

    def benchmark_inference(
        self, 
        text: str = "What is this?", 
        image_path: str = 'assets/image.png', 
        max_new_tokens: int = 100,
        num_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Perform comprehensive VLM inference benchmarking
        
        Args:
            text (str): Prompt text
            image_path (str): Path to input image
            max_new_tokens (int): Maximum number of tokens to generate
            num_runs (int): Number of benchmark runs
        
        Returns:
            Dict containing detailed benchmarking metrics
        """
        # Validate image path
        if not os.path.exists(image_path):
            print(f"Error: Image path {image_path} does not exist")
            print("Possible solutions:")
            print("1. Verify the correct path to your input image")
            print("2. Check current working directory")
            print(f"Current working directory: {os.getcwd()}")
            sys.exit(1)

        # Prepare input
        template = f"Question: {text} Answer:"
        encoded_batch = self.tokenizer.batch_encode_plus([template], return_tensors="pt")
        tokens = encoded_batch['input_ids'].to(self.device)
        
        # Calculate number of input tokens (for prefill tokens per second)
        input_token_count = tokens.shape[1]

        # Process image
        image = Image.open(image_path)
        image = self.image_processor(image)
        image = image.unsqueeze(0).to(self.device)

        # Benchmark results storage
        total_inference_times = []
        ttft_times = []
        total_tokens_generated = []

        for _ in range(num_runs):
            # Track time to first token (TTFT) - using explicit model call
            with torch.no_grad():
                # First measure TTFT with a single token generation
                first_token_start = time.time()
                
                # Generate just the first token
                try:
                    first_token_output = self.model.generate(
                        tokens, 
                        image, 
                        max_new_tokens=1
                    )
                except TypeError:
                    # Fallback to generation without image if needed
                    first_token_output = self.model.generate(
                        tokens, 
                        max_new_tokens=1
                    )
                
                first_token_time = time.time() - first_token_start
                
                # Reset for full generation
                start_time = time.time()
                
                # Generate all tokens
                try:
                    generated = self.model.generate(
                        tokens, 
                        image, 
                        max_new_tokens=max_new_tokens
                    )
                except TypeError:
                    # Fallback to generation without image if needed
                    generated = self.model.generate(
                        tokens, 
                        max_new_tokens=max_new_tokens
                    )
                
                total_inference_time = time.time() - start_time

            # Calculate tokens generated
            try:
                # Try to get sequence length directly
                num_tokens_generated = generated.shape[1] - tokens.shape[1]
            except (AttributeError, TypeError):
                # Fallback token counting
                try:
                    num_tokens_generated = len(generated) - tokens.shape[1]
                except:
                    # If all else fails, use a default
                    num_tokens_generated = max_new_tokens

            # Store metrics
            total_inference_times.append(total_inference_time)
            ttft_times.append(first_token_time)
            total_tokens_generated.append(num_tokens_generated)

        # Compute performance metrics
        metrics = {
            'avg_total_inference_time': sum(total_inference_times) / num_runs,
            'avg_time_to_first_token': sum(ttft_times) / num_runs,
            'avg_tokens_per_second': sum(total_tokens_generated) / sum(total_inference_times),
            'prefill_tokens_per_second': input_token_count / (sum(ttft_times) / num_runs) if sum(ttft_times) > 0 else 0,
            'tokens_generated_stats': {
                'min': min(total_tokens_generated),
                'max': max(total_tokens_generated),
                'avg': sum(total_tokens_generated) / num_runs
            }
        }

        return metrics

    def print_benchmark_results(self, metrics: Dict[str, Any]):
        """
        Pretty print benchmark results

        Args:
            metrics (Dict): Benchmarking metrics dictionary
        """
        print("\n--- VLM Inference Benchmark Results ---")
        print(f"Average Total Inference Time: {metrics['avg_total_inference_time']:.4f} seconds")
        print(f"Average Time to First Token (TTFT): {metrics['avg_time_to_first_token']:.4f} seconds")
        print(f"Average Tokens per Second: {metrics['avg_tokens_per_second']:.2f} TPS")
        print(f"Prefill Tokens per Second: {metrics['prefill_tokens_per_second']:.2f} tokens/s")


def main():
    # Detailed error handling for main execution
    try:
        # Create benchmark instance
        vlm_benchmark = VLMBenchmark()

        # Run benchmarking
        results = vlm_benchmark.benchmark_inference(
            text="What is in this image?", 
            image_path='assets/image.png', 
            max_new_tokens=100, 
            num_runs=10
        )

        # Print results
        vlm_benchmark.print_benchmark_results(results)

    except Exception as e:
        print(f"Unexpected error during benchmarking: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()