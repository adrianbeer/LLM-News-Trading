
import pandas as pd
from src.config import config, MODEL_CONFIG
import numpy as np
from typing import List
from multiprocessing.pool import ThreadPool
from functools import partial
import torch
import os
from torch import Tensor
from src.utils.time import timing
from transformers import BertTokenizerFast
from tqdm import tqdm


MAX_ENCODING_LENGTH = 512
DATASET_PATH = config.data.benzinga.cleaned

def embed_input(text, tokenizer):
    # Truncation = True as bert can only take inputs of max 512 tokens.
    # return_tensors = "pt" makes the funciton return PyTorch tensors
    # tokenizer.encode_plus specifically returns a dictionary of values instead of just a list of values
    encoding = tokenizer(
        text, 
        add_special_tokens = True, 
        truncation = True, 
        padding = "max_length", 
        max_length = MAX_ENCODING_LENGTH,
        return_attention_mask = True, 
        return_tensors = "pt"
    )
    # input_ids: mapping the words to tokens
    # attention masks: idicates if index is word or padding
    input_ids = encoding['input_ids']
    attention_masks = encoding['attention_mask']
    return input_ids, attention_masks


@timing
def embed_inputs(texts: list, tokenizer) -> tuple[Tensor, Tensor]:
    input_ids = []
    attention_masks = []
    
    print("Start embedding inputs...")
    executor = ThreadPool(processes=os.cpu_count())
    ans = tqdm(executor.imap(partial(embed_input, 
                                    tokenizer=tokenizer), 
                            texts),
               total=len(texts))
    input_ids, attention_masks = list(zip(*ans))

    input_ids: Tensor = torch.cat(input_ids, dim=0)
    attention_masks: Tensor = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks


def get_text_and_labels(dat: pd.DataFrame, 
                        text_col: str = None,
                        label_col: str = None) -> tuple[List, List]:
    texts = dat.loc[:, text_col].tolist()
    labels = dat.loc[:, label_col].tolist()
    return texts, labels


def get_encoding(encoding_matrix_path: str):
    encoding_matrix = np.load(file=encoding_matrix_path)
    index = encoding_matrix[:, 0]
    input_ids = encoding_matrix[:, 1:(MAX_ENCODING_LENGTH+1)]
    masks = encoding_matrix[:, (MAX_ENCODING_LENGTH+1):]
    return index, input_ids, masks


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_CONFIG.transformer_hugface_id)
    dataset = pd.read_parquet(DATASET_PATH)

    # Dummy column
    dataset["text_length"] = dataset["parsed_body"].map(lambda x: len(x))

    texts, labels = get_text_and_labels(dat=dataset, 
                                        text_col="parsed_body", 
                                        label_col="text_length")
    input_ids, masks = embed_inputs(texts, tokenizer)

    input_ids = pd.DataFrame(data=Tensor.numpy(input_ids), index=dataset.index)
    masks = pd.DataFrame(data=Tensor.numpy(masks), index=dataset.index)

    input_ids.columns = [str(x) for x in input_ids.columns]
    masks.columns = [str(x) for x in masks.columns]

    input_ids.to_parquet(config.data.benzinga.input_ids)
    masks.to_parquet(config.data.benzinga.masks)