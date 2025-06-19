import torch
from torch.utils.data import IterableDataset, get_worker_info
import threading
from queue import Queue
from typing import Iterator
import itertools


class ConstantLengthDataset(IterableDataset):
    def __init__(
        self,
        dataset, infinite: bool = False, seq_length: int = 1024, num_of_sequences: int = 1024, queue_size: int = 2048,
    ):
        self.dataset = dataset
        self.seq_length = seq_length
        self.max_length = seq_length * num_of_sequences
        self.epoch = 0  # only advanced when infinite=True
        self.infinite = infinite
        self.queue_size = queue_size
        self._sentinel = object()

    def __len__(self):
        return int(len(self.dataset) * 256/self.seq_length)

    def __iter__(self) -> Iterator[dict]:
        """
        Returns a consumer iterator that drains `self._queue`.
        A background thread keeps that queue topped up.
        """
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        def make_base_iterator():
            """Return a (sharded) iterator over the underlying dataset."""
            all_indices = range(len(self.dataset))

            # Shard the *indices* first, before any data is fetched.
            if num_workers > 1:
                worker_indices = itertools.islice(all_indices, worker_id, None, num_workers)
            else:
                worker_indices = all_indices

            # Create an iterator that only calls __getitem__ for the assigned indices.
            def sharded_item_iterator():
                for idx in worker_indices:
                    yield self.dataset[idx]

            return sharded_item_iterator()        
        
        queue: Queue = Queue(maxsize=self.queue_size)

        producer = threading.Thread(
            target=self._producer, args=(make_base_iterator, queue), daemon=True
        )
        producer.start()

        while True:
            sample = queue.get()
            if sample is self._sentinel:
                break
            yield sample

    def _producer(
            self,
            make_iterator,  # a zero-arg lambda that returns a fresh (possibly sharded) iterator
            queue: Queue,
        ):
        """Runs in a separate daemon thread and keeps `queue` full."""
        iterator = make_iterator()
        more_examples = True

        while more_examples:
            # ------------- 1) pull raw samples until we have enough -------- #
            buffer, buffer_len = [], 0
            while buffer_len < self.max_length:
                try:
                    sample = next(iterator)
                except StopIteration:
                    if self.infinite:
                        iterator = make_iterator()
                        self.epoch += 1
                        print(f"Epoch {self.epoch} finished, restarting iterator")
                        continue
                    else:
                        more_examples = False
                        break

                if len(sample["input_ids"]) > self.seq_length:
                    continue  # skip overly long samples

                buffer.append(sample)
                buffer_len += len(sample["input_ids"])

            if not buffer:
                break  # nothing left and not infinite

            # ------------- 2) run greedy knapsack & pack groups ------------ #
            lengths = [len(x["input_ids"]) for x in buffer]
            groups = self._balanced_greedy_knapsack(lengths, self.seq_length, delta=5)

            for g in groups:
                packed = self._pack_one_group(g, buffer, self.seq_length)

                # put blocks if queue is full.
                queue.put(
                    {
                        "input_ids": packed[0],
                        "labels": packed[1],
                        "attention_mask": packed[2],
                        "images": packed[3],
                    }
                )

        # finished → unblock consumer
        queue.put(self._sentinel)

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
