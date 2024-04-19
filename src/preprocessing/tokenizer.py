
from argparse import ArgumentParser
import pandas as pd
from src.config import config, PREP_CONFIG
import numpy as np
from typing import List
from multiprocessing.pool import ThreadPool
from functools import partial
import torch
import os
from torch import Tensor
from src.utils.time import timing
from transformers import AutoTokenizer
from tqdm import tqdm


DATASET_PATH = config.data.merged

def tokenize_input(text, tokenizer, max_encoding_length):
    # Truncation = True as bert can only take inputs of max 512 tokens.
    # return_tensors = "pt" makes the function return PyTorch tensors
    # tokenizer.encode_plus specifically returns a dictionary of values instead of just a list of values
    encoding = tokenizer(
        text, 
        add_special_tokens = True, 
        truncation = True, 
        padding = "max_length", 
        max_length = max_encoding_length,
        return_attention_mask = True, 
        return_tensors = "pt"
    )
    # input_ids: mapping the words to tokens
    # attention masks: idicates if index is word or padding
    input_ids = encoding['input_ids']
    attention_masks = encoding['attention_mask']
    return input_ids, attention_masks


@timing
def tokenize_inputs(texts: list, tokenizer, max_encoding_length: int) -> tuple[Tensor, Tensor]:
    input_ids = []
    attention_masks = []
    
    print("Start embedding inputs...")
    executor = ThreadPool(processes=os.cpu_count())
    ans = tqdm(executor.imap(partial(tokenize_input, 
                                    tokenizer=tokenizer,
                                    max_encoding_length=max_encoding_length), 
                            texts),
               total=len(texts))
    input_ids, attention_masks = list(zip(*ans))

    input_ids: Tensor = torch.cat(input_ids, dim=0)
    attention_masks: Tensor = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--title_only", action='store_true')
    args = parser.parse_args()

    print(f"Tokenizing the {DATASET_PATH=} using tokenizer {PREP_CONFIG.tokenizer=}")

    if args.title_only:
        text_col = "title" 
        input_ids_path = config.data.news.title_only.input_ids
        masks_path = config.data.news.title_only.masks
        max_encoding_length = 32
    else:
        text_col = "parsed_body"
        input_ids_path = config.data.news.input_ids
        masks_path = config.data.news.masks
        max_encoding_length = 256
    
    tokenizer = AutoTokenizer.from_pretrained(PREP_CONFIG.tokenizer)
    dataset = pd.read_parquet(DATASET_PATH, columns=[text_col])

    # Dummy column
    dataset["text_length"] = dataset[text_col].map(lambda x: len(x))

    texts = dataset.loc[:, text_col].tolist()
    input_ids, masks = tokenize_inputs(texts, tokenizer, max_encoding_length)

    input_ids = pd.DataFrame(data=Tensor.numpy(input_ids), index=dataset.index)
    masks = pd.DataFrame(data=Tensor.numpy(masks), index=dataset.index)

    input_ids.columns = [str(x) for x in input_ids.columns]
    masks.columns = [str(x) for x in masks.columns]

    input_ids.to_parquet(input_ids_path)
    masks.to_parquet(masks_path)
