import os
import torch
from PIL import Image
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

import models.config as config
from models.vision_language_model import VisionLanguageModel

from data.processors import get_image_processor, get_tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

vlm_cfg = config.VLMConfig()
train_cfg = config.TrainConfig()

image_processor = get_image_processor(vlm_cfg.vit_img_size)
tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)


print(train_cfg.train_dataset_name[0])
train_ds = load_dataset(
    train_cfg.train_dataset_path, train_cfg.train_dataset_name[0], split="train"
)
test_ds = load_dataset(train_cfg.test_dataset_path, split ="val")

print("---------------------")
print(train_ds)
print(test_ds)

print("---------------------")
print("process test dataset to have similar format to train dataset")
# rename image column to images 
test_ds = test_ds.rename_column("image", "images")
# combine question and answer columns into a column called texts 
test_ds = test_ds.map(
    lambda x: {"texts": [{"user": x["question"],"assistant": x["answer"]}]},
    # tidying up is good for the environment
    remove_columns=["question", "answer","index","category","l2_category","meta_info"], 
)


print(test_ds) # now has only 2 columns images and texts
print(test_ds[0])
print("---------------------")




def data_collator(examples, tokenizer=tokenizer, image_processor=image_processor):
    "collate and process the data 2 in one"
    # image
    images = []
    for example in examples:
        # single image per example
        image = example["images"]
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            processed_image = image_processor(image)
        else:
            # Create empty tensor with right dimensions as fallback
            processed_image = torch.zeros(3, vlm_cfg.vit_img_size, vlm_cfg.vit_img_size)

        images.append(processed_image)
        # process the texts using the tokenizer

    # stack the images into tensors
    images = torch.stack(images)

    # text
    input_sequences = []
    for example in examples:
        txt = example["texts"][0]
        # following the implementation in the datasets.py file
        texts = f"Question: {txt['user']} Answer: "
        answers = txt["assistant"]
        input_sequences.append(f"{texts} {answers}")
    
    encoded_full_sequences = tokenizer.batch_encode_plus(
            input_sequences,
            padding="longest", # better than "max_length" 
            padding_side="right",
            return_tensors="pt",
            truncation=True,
            max_length=vlm_cfg.lm_max_length,
        )
    
    input_ids = encoded_full_sequences["input_ids"]
    attention_mask = encoded_full_sequences["attention_mask"]
    
    # better than the original implementation because we don't add an extra pad token
    # see VAQCollator.__call__ for more details
    input_ids, labels = input_ids[:,:-1], input_ids[:, 1:].clone() # different tensors because we'll update the labels
    attention_mask = attention_mask[:, :-1]
    
    # update labels to be -100 where the attention mask is 0
    # using attention mask shifted to keep the first eos token
    labels[attention_mask == 0] = -100
    # no need to update the input_ids since we are going to skip loss calculation for those tokens
    
    # will be passed as **kwargs to the forward method
    return {
        "input_ids": input_ids,
        "image": images,
        "attention_mask": attention_mask,
        "labels": labels, # following exact name in forward method (would have been better if it was labels)
    }


model= VisionLanguageModel(vlm_cfg)

# import code
# code.interact(local=locals())

args = TrainingArguments(
    output_dir="output",
    max_steps=20,
    per_device_train_batch_size= 2, # still locally rn :)
    per_device_eval_batch_size=2,
    logging_steps=10,
    eval_strategy="steps",
    remove_unused_columns=False,
    # label_names= ["labels"], 
    # solve safetensors issue
    save_safetensors =False,
    
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=data_collator,
)
trainer.train()