import torch

class VQACollator(object):  # Visual Question Answering Collator
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch):
        images = [item["image"] for item in batch]
        texts = [item["text_data"] for item in batch]
        answers = [item["answer"] for item in batch]

        # Stack images
        images = torch.stack(images)

        # Create inputs by concatenating the question and answer
        input_sequences = []
        for i in range(len(texts)):
            input_sequences.append(f"{texts[i]}{answers[i]}")

        encoded_full_sequences = self.tokenizer.batch_encode_plus(
            input_sequences,
            padding="max_length",
            padding_side="left",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        # Create labels where only answer tokens are predicted
        input_ids = encoded_full_sequences["input_ids"]
        attention_mask = encoded_full_sequences["attention_mask"]
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:].clone()
        labels[:, -1] = -100 #self.tokenizer.pad_token_id

        # The tokenizer has different behavior for padding and truncation:
        # 1. If the full text (answer + question) is shorter than the max length, it gets padded on the left
        # 2. If the full text is longer than the max length, it gets truncated on the right
        # Therefore, I need to handle multiple cases, this is the different scenarios:
        # If the full text is longer than the max length, we need to set the labels to -100 for the whole sample (we want to ignore the whole sample)
        # If the full text is shorter than the max length, we need to set the labels to -100 only for the question part, and create causal language modeling labels for the answer part, taking into account the padding

        # Determine if sequences were truncated
        original_lengths = [len(self.tokenizer.encode(seq)) for seq in input_sequences]
        
        for i in range(len(batch)):
            # Get the length of the question for this sample
            question_length = len(self.tokenizer.encode(texts[i], add_special_tokens=False))
            
            # Case 1: If sequence was truncated (original is longer than max_length)
            if original_lengths[i] > self.max_length:
                # Set all labels to -100 to ignore this sample entirely
                labels[i, :] = -100
                #print(f"Sample {i} was truncated. Setting all labels to -100.")
                continue
            
            # Case 2: Sequence fits within max_length
            # Use attention mask to find first non-padding token
            # The first 1 in the attention mask marks the first non-padding token
            first_token_pos = attention_mask[i].nonzero(as_tuple=True)[0][0].item()
            
            # Set labels for padding and question part to -100 (don't predict these), substracting 1 to account for the left shift
            question_end = first_token_pos + question_length - 1 
            labels[i, :question_end] = -100
            # labels[i, original_lengths[i]-1:] = -100 # If you are using right padding

        return {
            "image": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class MMStarCollator(object):  # https://huggingface.co/datasets/Lin-Chen/MMStar
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        images = [item["image"] for item in batch]
        questions = [item["text_data"] for item in batch]
        answers = [item["answer"] for item in batch]

        # Stack images
        images = torch.stack(images)
        
        encoded_question_sequences = self.tokenizer.batch_encode_plus(
            questions,
            padding=True,
            padding_side="left",
            return_tensors="pt"
        )

        encoded_answer_sequences = self.tokenizer.batch_encode_plus(
            answers,
            padding=True,
            padding_side="left",
            return_tensors="pt"
        )
        
        return {
            "images": images,
            "input_ids": encoded_question_sequences['input_ids'],
            "attention_mask": encoded_question_sequences['attention_mask'],
            "labels": encoded_answer_sequences['input_ids'],
        }