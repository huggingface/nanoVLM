import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset


class BaseDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length

        self.prefix_len = self._get_prefix_len()

    def __len__(self):
        return len(self.dataset)

    def _get_prefix_len(self):
        random_string_5_letters = "xzyvd"
        random_string_chat_templated = self.tokenizer.apply_chat_template([{"role": "assistant", "content": random_string_5_letters}], tokenize=False, add_special_tokens=False)
        random_string_location = random_string_chat_templated.find(random_string_5_letters)
        return len(self.tokenizer.encode(random_string_chat_templated[:random_string_location]))

    def _get_messages(self, item, image_count=0):
        messages = []
        for text in item['texts']:
            messages.append({"role": "user", "content": text['user']})
            messages.append({"role": "assistant", "content": text['assistant']})

        if image_count > 0:
            messages[0]["content"] = self.tokenizer.image_token * image_count * self.mp_image_token_length + messages[0]["content"]      

        return messages

    def _process_images(self, images):
        processed_images = []
        for image in images:
            if isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                processed_image = self.image_processor(image)
                processed_images.append(processed_image)
            else:
                raise ValueError(f"Error processing image")
        return processed_images


    def _prepare_inputs_and_loss_mask(self, messages):
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        mask = [0] * len(conv_ids["input_ids"])

        # Locate each assistant turn and flip its mask to 1
        cursor = 0
        for msg in messages:
            segment_ids = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_special_tokens=False
            )
            seg_len = len(segment_ids)

            if msg["role"] == "assistant":
                start = cursor + self.prefix_len
                end   = cursor + seg_len
                mask[start:end] = [1] * (end - start)  # attend to these tokens

            cursor += seg_len
        
        return torch.tensor(conv_ids["input_ids"]), torch.tensor(mask).to(torch.bool), torch.tensor(conv_ids["attention_mask"])


class VQADataset(BaseDataset):  # Visual Question Answering Dataset
    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Handle images (should be a list)
        images_data = item['images']
        if not isinstance(images_data, list):
            images_data = [images_data]

        # Now process the images
        processed_images = self._process_images(images_data)

        messages = self._get_messages(item, len(processed_images))

        input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, mask)

        return {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _get_labels(self, input_ids, mask):
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1) # Shift labels for causal LM
        labels[-1] = -100 # Last token has no target
        
        return labels

class MMStarDataset(BaseDataset):  # https://huggingface.co/datasets/Lin-Chen/MMStar
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        image = item['image']
        processed_images = self._process_images([image])
        
        item['texts'] = [{
            "user": item['question'] +  "\nAnswer only with the letter!",
            "assistant": item['answer']
        }]
        messages = self._get_messages(item, image_count=len(processed_images))

        input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, mask)
        input_ids = input_ids.masked_fill(mask, self.tokenizer.pad_token_id)
        attention_mask = attention_mask.masked_fill(mask, 0)

        return {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _get_labels(self, input_ids, mask):
        labels = input_ids.clone().masked_fill(~mask, self.tokenizer.pad_token_id)
        return labels


class ConstantLengthDataset(IterableDataset):
    def __init__(
        self, dataset, infinite=False, seq_length=1024, num_of_sequences=1024
    ):
        self.dataset = dataset
        self.seq_length = seq_length
        self.max_length = seq_length * num_of_sequences
        self.epoch = 0
        self.infinite = infinite

    def __len__(self):
        return int(len(self.dataset) * 256/self.seq_length)

    def _balanced_greedy_knapsack(self, lengths, L, delta=0):
        # keep the position while sorting
        items = sorted(enumerate(lengths), key=lambda x: x[1], reverse=True)

        min_knapsacks   = (sum(lengths) + L - 1) // L + delta
        knapsack_load   = [0]   * min_knapsacks
        knapsack_groups = [[] for _ in range(min_knapsacks)]

        for idx, item_len in items:
            # choose the lightest knapsack so far
            ks_id = min(range(len(knapsack_load)),
                        key=knapsack_load.__getitem__)

            # open a new one if the chosen bag would overflow
            if knapsack_load[ks_id] + item_len > L:
                ks_id = len(knapsack_load)
                knapsack_load   .append(0)
                knapsack_groups.append([])

            knapsack_groups[ks_id].append(idx)
            knapsack_load  [ks_id] += item_len

        # remove the completely empty bags that the +delta heuristic created
        return [g for g in knapsack_groups if g]

    def _pack_one_group(self, group_indices, batch,
                        max_len):
        ids, lbl, am, ims = [], [], [], []

        for i in group_indices:
            ids.extend(batch[i]["input_ids"])
            lbl.extend(batch[i]["labels"])
            am .extend(batch[i]["attention_mask"])
            ims.extend(batch[i]["images"])

        # safety: assert we never overflow
        if len(ids) > max_len:
            raise ValueError(f"Packed length {len(ids)} > max_len {max_len}")

        return torch.stack(ids), torch.stack(lbl), torch.stack(am), ims

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_length:
                    break
                try:
                    next_sample = next(iterator)
                    if len(next_sample["input_ids"]) > self.seq_length:
                        continue  # Discard samples that are too long
                    buffer.append(next_sample)
                    buffer_len += len(next_sample["input_ids"])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                    else:
                        more_examples = False
                        break

            # 1) run knapsack on the lengths
            lengths = list(map(lambda x: len(x["input_ids"]), buffer))
            groups  = self._balanced_greedy_knapsack(lengths, self.seq_length, delta=5)

            # 2) build new (packed) lists
            packed_ids, packed_lbl, packed_am, packed_imgs = [], [], [], []

            for g in groups:
                ids, lbl, am, ims = self._pack_one_group(
                    g, buffer, self.seq_length
                )
                packed_ids .append(ids)
                packed_lbl .append(lbl)
                packed_am  .append(am)
                packed_imgs.append(ims)

            for i in range(len(packed_ids)):
                yield {
                    "input_ids": packed_ids[i],
                    "labels": packed_lbl[i],
                    "attention_mask": packed_am[i],
                    "images": packed_imgs[i]
                }
