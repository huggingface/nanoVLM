import torch
import argparse
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from data.collators import VQACollator
from data.datasets import VQADataset
from data.processors import get_image_processor, get_tokenizer
from models.vision_language_model import VisionLanguageModel
import models.config as config

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def measure_vram(args, vlm_cfg, train_cfg_defaults):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available. VRAM measurement requires a CUDA-enabled GPU.")
        return

    # --- Model Initialization ---
    torch.cuda.reset_peak_memory_stats(device)
    print(f"Using VLMConfig defaults: load_backbone_weights={vlm_cfg.vlm_load_backbone_weights}")
    model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)

    if args.compile:
        print("Compiling the model with torch.compile...")
        model = torch.compile(model)
        print("Model compiled.")
    
    model.to(device)

    # Measure VRAM after model is loaded to device
    torch.cuda.synchronize() # Ensure all operations are complete
    initial_vram_allocated_bytes = torch.cuda.memory_allocated(device)
    initial_vram_allocated_mb = initial_vram_allocated_bytes / (1024 ** 2)
    print(f"VRAM allocated after loading model to device: {initial_vram_allocated_mb:.2f} MB")

    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # --- Dataset Preparation ---
    image_processor = get_image_processor(vlm_cfg.vit_img_size)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)

    dataset_path = train_cfg_defaults.train_dataset_path
    # train_cfg_defaults.train_dataset_name is a list, use the first if not specified
    dataset_name = train_cfg_defaults.train_dataset_name[0] if train_cfg_defaults.train_dataset_name else None

    batch_sizes_to_test = [int(bs) for bs in args.batch_sizes.split()]
    if not batch_sizes_to_test:
        print("Error: No batch sizes provided or parsed correctly.")
        return
    
    num_iterations_for_vram = args.num_iterations
    max_bs_to_test = max(batch_sizes_to_test)
    required_samples_for_base_ds = max_bs_to_test * num_iterations_for_vram

    try:
        print(f"Loading dataset: {dataset_path}, name: {dataset_name}")
        # Attempt to load only the 'train' split, adjust if dataset has different split names
        available_splits = load_dataset(dataset_path, dataset_name).keys()
        split_to_use = 'train' if 'train' in available_splits else list(available_splits)[0]
        
        base_ds_full = load_dataset(dataset_path, dataset_name, split=split_to_use)
        
        if len(base_ds_full) < required_samples_for_base_ds:
            print(f"Warning: Dataset '{dataset_name}' (split: {split_to_use}) has {len(base_ds_full)} samples, "
                  f"but {required_samples_for_base_ds} are recommended for max batch size {max_bs_to_test} "
                  f"and {num_iterations_for_vram} iterations. Using all available samples.")
            base_ds_for_vram_test = base_ds_full
        else:
            base_ds_for_vram_test = base_ds_full.select(range(required_samples_for_base_ds))
        print(f"Using {len(base_ds_for_vram_test)} samples for VRAM testing.")
    except Exception as e:
        print(f"Error loading dataset: {dataset_path}, name: {dataset_name}. Error: {e}")
        print("Please ensure the dataset path and name are correct.")
        return

    processed_base_dataset = VQADataset(base_ds_for_vram_test, tokenizer, image_processor)
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)

    print("\n--- VRAM Measurement ---")
    results = {}

    for bs in batch_sizes_to_test:
        print(f"\nTesting Batch Size: {bs}")
        
        if len(processed_base_dataset) < bs:
            print(f"Base processed dataset has {len(processed_base_dataset)} samples, "
                  f"not enough for batch size {bs}. Skipping.")
            results[bs] = "Not enough data"
            continue

        current_loader = DataLoader(
            processed_base_dataset,
            batch_size=bs,
            shuffle=False, 
            collate_fn=vqa_collator,
            num_workers=0,
            pin_memory=True,
            drop_last=True # Important if dataset size is not exactly multiple of bs
        )

        if len(current_loader) < num_iterations_for_vram:
             print(f"Dataloader for batch size {bs} yields {len(current_loader)} batches, "
                   f"less than requested {num_iterations_for_vram} iterations. Will run available batches.")
             if len(current_loader) == 0:
                 print(f"Dataloader for batch size {bs} is empty. Skipping.")
                 results[bs] = "Dataloader empty"
                 continue


        # Reset CUDA memory stats for each batch size test
        torch.cuda.reset_peak_memory_stats(device)
        
        # Model to train mode for realistic scenario (e.g. dropout layers active)
        model.train() 
        optimizer = optim.AdamW(model.parameters(), lr=1e-5) # Dummy optimizer

        try:
            for i, batch in enumerate(current_loader):
                if i >= num_iterations_for_vram:
                    break
                
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16): # Doing autocast to stay close the train.py script
                    _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
                
                if loss is not None:
                    loss.backward()
                    optimizer.step()
                else:
                    print("Warning: Model did not return loss. Backward pass and optimizer step skipped. VRAM for these operations will not be measured.")

            peak_vram_allocated_bytes = torch.cuda.max_memory_allocated(device)
            peak_vram_allocated_mb = peak_vram_allocated_bytes / (1024 ** 2)
            print(f"Peak VRAM allocated for batch size {bs}: {peak_vram_allocated_mb:.2f} MB")
            results[bs] = f"{peak_vram_allocated_mb:.2f} MB"

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                peak_vram_allocated_bytes = torch.cuda.max_memory_allocated(device) # Get max allocated before OOM
                peak_vram_allocated_mb = peak_vram_allocated_bytes / (1024 ** 2)
                print(f"CUDA out of memory for batch size {bs}. ")
                print(f"Peak VRAM allocated before OOM: {peak_vram_allocated_mb:.2f} MB (may be approximate)")
                results[bs] = f"OOM (Peak before OOM: {peak_vram_allocated_mb:.2f} MB)"
            else:
                print(f"An unexpected runtime error occurred for batch size {bs}: {e}")
                results[bs] = f"Error: {e}"
                # raise e # Optionally re-raise for debugging
        finally:
            del current_loader, optimizer
            if 'loss' in locals() and loss is not None : del loss
            if 'images' in locals(): del images
            if 'input_ids' in locals(): del input_ids
            if 'labels' in locals(): del labels
            if 'attention_mask' in locals(): del attention_mask
            torch.cuda.empty_cache()
    
    print("\n--- Summary of VRAM Usage ---")
    for bs, vram_usage in results.items():
        print(f"Batch Size {bs}: {vram_usage}")


def main():
    parser = argparse.ArgumentParser(description="Measure VRAM usage for a VisionLanguageModel at different batch sizes.")
    
    # Model and Config args
    parser.add_argument('--compile', action='store_true', help='Compile the model with torch.compile.')

    # Measurement control args
    parser.add_argument('--batch_sizes', type=str, default="1 2 4", help='Space-separated list of batch sizes to test (e.g., "1 2 4 8").')
    parser.add_argument('--num_iterations', type=int, default=2, help='Number of forward/backward passes per batch size for VRAM measurement.')

    args = parser.parse_args()

    vlm_cfg = config.VLMConfig()
    train_cfg_defaults = config.TrainConfig() # Used for default dataset path/name if not provided by CLI

    print("--- VLM Config (from models.config) ---")
    print(vlm_cfg) # Show base config
    print("--- Train Config Defaults (for dataset path/name if not specified via CLI) ---")
    print(f"Default dataset_path: {train_cfg_defaults.train_dataset_path}")
    print(f"Default dataset_name list: {train_cfg_defaults.train_dataset_name}")
    
    measure_vram(args, vlm_cfg, train_cfg_defaults)

if __name__ == "__main__":
    main() 