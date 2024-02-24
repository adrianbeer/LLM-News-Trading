# Calculate staleness factor of news
import pandas as pd
import torch
from transformers import BertModel
from src.model.data_loading import create_dataloader
from src.config import config, MODEL_CONFIG
from numpy import dot
from numpy.linalg import norm
from src.model.neural_network import predict_cls
from transformers.models.bert.modeling_bert import BertModel
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', help='batch size')
parser.add_argument('--generate_cls_tokens', action='store_true', help='')
parser.add_argument('--calculate_staleness', action='store_true', help='')


if __name__ == '__main__':
    args = parser.parse_args()
    batch_size = int(args.batchsize)
    
    if args.generate_cls_tokens:
        print("Start loading in the BERT model...")
        # Use baseline bert model to avoid look-ahead bias 
        model = BertModel.from_pretrained(MODEL_CONFIG.pretrained_network)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model.to(device)
        
        dataset = pd.read_parquet(config.data.news.cleaned, columns=["stocks"])
        input_ids = pd.read_parquet(config.data.news.input_ids)
        masks = pd.read_parquet(config.data.news.masks)
        
        if not dataset.index.name: 
            dataset.index.name = "index"
            
        original_index_name = dataset.index.name
        print(f"{original_index_name=}")

        # Align DataFrames
        input_ids = input_ids.loc[dataset.index]
        masks = masks.loc[dataset.index]

        # Convert to Tensors
        input_ids = torch.from_numpy(input_ids.to_numpy())
        masks = input_ids = torch.from_numpy(masks.to_numpy())

        dataloader = create_dataloader(tensors=[input_ids, masks], 
                                    batch_size=batch_size, 
                                    data_loader_kwargs=dict(shuffle=False))
        cls_tokens = predict_cls(model, dataloader, device)
        cls_tokens = pd.Series(index=dataset.index, data=list(cls_tokens))
        cls_tokens.to_pickle("data/news/cls_tokens.pkl")
        print("Finished")
    
    if args.calculate_staleness:
        dataset = pd.read_parquet(config.data.news.cleaned)
        cls_tokens = pd.read_pickle("data/news/cls_tokens.pkl")
        dataset["cls_token"] = cls_tokens
        dataset["staleness"] = 0.0
        n_of_sametime_news = 0

        # To determine the freshness of news, I compare the similarity of each news article with all articles published in the previous three days.
        for ticker in tqdm(set(dataset.stocks), desc="stocks"):
            ticker_news = dataset[dataset.stocks == ticker].reset_index()
            ticker_news = ticker_news.set_index("time").sort_index(ascending=True)
            # Set staleness of first news message to 0 
            ticker_news.at[ticker_news.index[0], "staleness"] = 0

            for time in  ticker_news.index:
                previous_news = ticker_news.loc[(time-pd.DateOffset(days=3)):time, "cls_token"]
                if len(previous_news) == 1:
                    ticker_news.at[time, "staleness"] = 0
                else:
                    try:
                        current_cls = previous_news.pop(time)
                    except IndexError as e:
                        n_of_sametime_news += 1
                        current_cls = previous_news.iloc[-1]
                        previous_news = previous_news.iloc[:-1]
                        
                    cosine_sims = previous_news.apply(lambda x: dot(current_cls, x) / (norm(current_cls)*norm(x)))
                    ticker_news.at[time, "staleness"] = cosine_sims.max()
                    
            ticker_news.set_index(original_index_name, inplace=True)
            # Add entries to data set
            dataset.loc[ticker_news.index, "staleness"] = ticker_news.loc[:, "staleness"]
            
        print(n_of_sametime_news)
        dataset.to_parquet(config.data.news.cleaned)
        print("Finished")

