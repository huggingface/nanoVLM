import time
import torch
import wandb
import argparse
import torch.optim as optim
from statistics import mean
from dataclasses import asdict
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from data.collators import VQACollator, MMStarCollator
from data.datasets import MMStarDataset, VQADataset
from data.processors import get_image_processor, get_tokenizer
from models.vision_language_model import VisionLanguageModel
import models.config as config
import models.utils as utils

#Otherwise, the tokenizer will through a warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def init_dist():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

def destroy_dist():
    dist.destroy_process_group()

def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_master():
    return dist.get_rank() == 0 if is_dist() else True

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def get_rank():
    return dist.get_rank() if is_dist() else 0

def dist_gather(o):
    o_all = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(o_all, o)
    return o_all

def wrap_model(model):
    return DistributedDataParallel(model, device_ids=[dist.get_rank()])

def get_run_name(train_cfg):
    dataset_size = "full_ds" if train_cfg.data_cutoff_idx is None else f"{train_cfg.data_cutoff_idx}samples"
    batch_size = f"bs{train_cfg.batch_size}"
    epochs = f"ep{train_cfg.epochs}"
    learning_rate = f"lr{train_cfg.lr_backbones}-{train_cfg.lr_mp}"
    num_gpus = f"{get_world_size()}xGPU"
    date = time.strftime("%m%d")

    return f"nanoVLM_{num_gpus}_{dataset_size}_{batch_size}_{epochs}_{learning_rate}_{date}"

def get_dataloaders(train_cfg, vlm_cfg):
    # Create datasets
    image_processor = get_image_processor(vlm_cfg.vit_img_size)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)

    # Load and combine all training datasets
    combined_train_data = []
    for dataset_name in train_cfg.train_dataset_name:
        train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name)
        combined_train_data.append(train_ds['train'])
    train_ds = concatenate_datasets(combined_train_data)
    
    test_ds = load_dataset(train_cfg.test_dataset_path)

    # Apply cutoff if specified
    if train_cfg.data_cutoff_idx is None:
        total_samples = len(train_ds)  # Use the entire dataset
    else:
        total_samples = min(len(train_ds), train_cfg.data_cutoff_idx)

    train_dataset = VQADataset(train_ds.select(range(total_samples)), tokenizer, image_processor)
    test_dataset = MMStarDataset(test_ds['val'], tokenizer, image_processor)

    # Create collators
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    mmstar_collator = MMStarCollator(tokenizer)

    # Create dataloaders
    train_sampler = DistributedSampler(
        train_dataset, 
        rank=get_rank(),
        num_replicas=get_world_size(),
        shuffle = False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,    # =per device BS in DDP
        shuffle = False,
        sampler=train_sampler,
        collate_fn=vqa_collator,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=train_cfg.mmstar_batch_size, 
        shuffle=False, 
        collate_fn=mmstar_collator,
        pin_memory=True
    )

    return train_loader, test_loader

def test_mmstar(model, tokenizer, test_loader, device):
    model.eval()
    total_examples = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in test_loader:
            image = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            correct_answer = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            if is_dist():
                gen = model.module.generate(input_ids, image, attention_mask)
            else:
                gen = model.generate(input_ids, image, attention_mask)
            model_output = tokenizer.batch_decode(gen, skip_special_tokens=True)
            
            is_correct = utils.check_multiple_choice_with_regex(model_output, correct_answer)
            
            total_examples += len(is_correct)
            if is_correct:
                correct_predictions += sum(is_correct)

    accuracy = correct_predictions / total_examples if total_examples > 0 else 0

    return accuracy

def train(train_cfg, vlm_cfg):
    train_loader, test_loader = get_dataloaders(train_cfg, vlm_cfg)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)

    total_dataset_size = len(train_loader.dataset)
    if train_cfg.log_wandb and is_master():
        run_name = get_run_name(train_cfg)
        if train_cfg.data_cutoff_idx is None:
            run_name = run_name.replace("full_ds", f"{total_dataset_size}samples")
        run = wandb.init(
            entity=train_cfg.wandb_entity,
            project="nanoVLM",
            config={
                "VLMConfig": asdict(vlm_cfg),
                "TrainConfig": asdict(train_cfg)
            },
            name=run_name,
        )

    # Initialize model
    if train_cfg.resume_from_vlm_checkpoint:
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)
    
    if is_master():
        print(f"nanoVLM initialized with {sum(p.numel() for p in model.parameters()):,} parameters") 
        print(f"Training summary {'(global)' if is_dist() else ''}: {len(train_loader.dataset)} samples, {len(train_loader)*get_world_size()} batches/epoch, batch size {train_cfg.batch_size*get_world_size()}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")

        if is_dist():
            print(f"Training summary per GPU: {len(train_loader)} batches/epoch, batch size {train_loader.batch_size}")

    # Define optimizer groups
    # Since we have pretrained vision and language backbones, but a newly initialized modality projection layer, it doesn't make sense to train the with the same learning rate
    # You could opt to fully freeze the backbones and only train the MP layer, but finetuning them with a lower learning rate makes the training as a whole easier
    param_groups = [{'params': model.MP.parameters(), 'lr': train_cfg.lr_mp},
                    {'params': list(model.decoder.parameters()) + list(model.vision_encoder.parameters()), 'lr': train_cfg.lr_backbones}]
    optimizer = optim.AdamW(param_groups)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if train_cfg.compile:
        model = torch.compile(model)
    if is_dist():
        model = wrap_model(model)

    epoch_times = []
    best_accuracy = 0
    global_step = 0
    for epoch in range(train_cfg.epochs):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_tokens_processed = 0

        for batch in train_loader:
            batch_start_time = time.time()
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16): # Set to float16 if your hardware doesn't support bfloat16ß
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_train_loss += batch_loss

            num_tokens = torch.sum(attention_mask).item() # Sum of attention mask gives number of tokens
            num_tokens += images.shape[0] * ((images.shape[2] / vlm_cfg.vit_patch_size) ** 2) / (vlm_cfg.mp_pixel_shuffle_factor ** 2) # Add image tokens = batch_size * (((img_size / patch_size) ** 2) / (pixel_shuffle_factor ** 2))
            total_tokens_processed += num_tokens

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = num_tokens / batch_duration 

            # gather loss and t/s from all ranks if DDP
            batch_loss = mean(dist_gather(batch_loss)) if is_dist() else batch_loss  
            tokens_per_second = sum(dist_gather(tokens_per_second)) if is_dist() else tokens_per_second  

            if train_cfg.eval_in_epochs and global_step % 100 == 0 and is_master():
                epoch_accuracy = test_mmstar(model, tokenizer, test_loader, device)
                if epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    if is_dist(): 
                        model.module.save_pretrained(save_directory=vlm_cfg.vlm_checkpoint_path)
                    else:
                        model.save_pretrained(save_directory=vlm_cfg.vlm_checkpoint_path)
                    print(f"Step: {global_step}, Loss: {batch_loss:.4f}, Tokens/s: {tokens_per_second:.2f}, Accuracy: {epoch_accuracy:.4f} | Saving checkpoint to {vlm_cfg.vlm_checkpoint_path}")
                else:
                    print(f"Step: {global_step}, Loss: {batch_loss:.4f}, Tokens/s: {tokens_per_second:.2f}, Accuracy: {epoch_accuracy:.4f}")
                if train_cfg.log_wandb and is_master():
                    run.log({"accuracy": epoch_accuracy}, step=global_step)

            if train_cfg.log_wandb and is_master():
                run.log({"batch_loss": batch_loss,
                         "tokens_per_second": tokens_per_second}, step=global_step)

            global_step += 1

        avg_train_loss = total_train_loss / len(train_loader)
        # gather average batch loss from all ranks if DDP
        avg_train_loss = mean(dist_gather(avg_train_loss)) if is_dist() else avg_train_loss  

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # gather and sum total_tokens_processed accross all ranks if DDP
        total_tokens_processed = sum(dist_gather(total_tokens_processed)) if is_dist() else total_tokens_processed  
        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        if is_master():
            if train_cfg.log_wandb:
                run.log({"epoch_loss": avg_train_loss,
                         "epoch_duration": epoch_duration,
                         "epoch_tokens_per_second": epoch_tokens_per_second})

            print(f"Epoch {epoch+1}/{train_cfg.epochs}, Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}")

    # Summary Statistics
    if is_master():
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_training_time = sum(epoch_times)
        total_samples_processed = len(train_loader.dataset) * train_cfg.epochs
        avg_time_per_sample = total_training_time / total_samples_processed
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"Average time per sample: {avg_time_per_sample:.4f}s")

        # accuracy = test_mmstar(model, tokenizer, test_loader, device)
        # print(f"MMStar Accuracy: {accuracy:.4f}")

        # if train_cfg.log_wandb:
        #     run.summary["avg_epoch_time"] = avg_epoch_time
        #     run.summary["avg_time_per_sample"] = avg_time_per_sample
        #     run.summary["mmstar_acc"] = accuracy
        #     run.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_mp', type=float, help='Learning rate for the mapping network')
    parser.add_argument('--lr_backbones', type=float, help='Learning rate for the backbones')
    parser.add_argument('--vlm_checkpoint_path', type=str, help='Path to the VLM checkpoint')
    parser.add_argument('--resume_from_vlm_checkpoint', type=bool, default=False, help='Resume training from VLM checkpoint')

    args = parser.parse_args()

    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig()

    if args.lr_mp is not None:
        train_cfg.lr_mp = args.lr_mp
    if args.lr_backbones is not None:
        train_cfg.lr_backbones = args.lr_backbones
    if args.vlm_checkpoint_path is not None:
        vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path
    # Override resume flag based on whether a checkpoint path was provided or explicitly set
    if args.resume_from_vlm_checkpoint and args.vlm_checkpoint_path is not None:
         train_cfg.resume_from_vlm_checkpoint = True
         # Ensure loading flags are set correctly if resuming
         vlm_cfg.vlm_load_backbone_weights = False

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()
        if is_master():
            print(f"--- DDP Training Run (world size: {dist.get_world_size()}) ---")

    if is_master():
        print("--- VLM Config ---")
        print(vlm_cfg)
        print("--- Train Config ---")
        print(train_cfg)

    train(train_cfg, vlm_cfg)

    if is_dist():
        destroy_dist()

if __name__ == "__main__":
    main()