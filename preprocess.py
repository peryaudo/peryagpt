#!/usr/bin/env python
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

dataset = load_dataset("wikipedia", "20220301.en")

dataset = dataset.shuffle()
dataset = dataset['train'].train_test_split(test_size=0.01)

def process(example):
    tokens = np.frombuffer(example["text"].encode("utf-8"), dtype=np.uint8)
    return {"tokens": tokens, "len": len(tokens)}

processed_dataset = dataset.map(process, num_proc=8)

def write_dataset(filename, processed_dataset):
    array_len = np.sum(processed_dataset["len"], dtype=np.uint64)
    array = np.memmap(filename, dtype=np.uint8, mode="w+", shape=(array_len,))

    idx = 0
    for example in tqdm(processed_dataset):
        array[idx : (idx + example["len"])] = example["tokens"]
        idx += example["len"]
    array.flush()

write_dataset("val.bin", processed_dataset['test'])
write_dataset("train.bin", processed_dataset['train'])