import torch
from PIL import Image
from torch.utils.data import Dataset

import models.config as cfg


class VQADataset(Dataset):  # Visual Question Answering Dataset
    def __init__(self, dataset, tokenizer, image_processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Handle image (it's a list)
        image_data = item['images']
        if isinstance(image_data, list) and len(image_data) > 0:
            image = image_data[0]
        else:
            image = image_data

        # Now process the image
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            processed_image = self.image_processor(image)
        else:
            print(f"Error processing image at index {idx}")
            # Create empty tensor with right dimensions as fallback
            processed_image = torch.zeros(
                3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)

        # Process text (also a list)
        text_data = item['texts']
        if isinstance(text_data, list) and len(text_data) > 0:
            text = text_data[0]
        else:
            text = text_data

        question = text['user']
        # Add EOS token to the answer to train model to predict it, enabling correct stopping during generation
        answer = text['assistant'] + self.tokenizer.eos_token

        formatted_text = f"Question: {question} Answer:"

        return {
            "image": processed_image,
            "text_data": formatted_text,
            "answer": answer
        }


class MMStarDataset(Dataset):  # https://huggingface.co/datasets/Lin-Chen/MMStar
    def __init__(self, dataset, tokenizer, image_processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        image = item['image']
            
        # Now process the image
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            processed_image = self.image_processor(image)
        else:
            print(f"Error processing image at index {idx}")
            # Create empty tensor with right dimensions as fallback
            processed_image = torch.zeros(3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)
        
        question = item['question']
        answer = item['answer'] + self.tokenizer.eos_token # Add EOS token to the answer to train model to predict it, enabling correct stopping during generation
        
        formatted_text = f"Question: {question} \nAnswer only with the letter! \nAnswer:"
        
        return {
            "image": processed_image,
            "text_data": formatted_text,
            "answer": answer
        }
    