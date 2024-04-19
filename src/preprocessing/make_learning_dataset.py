
import pandas as pd
from src.config import config, PREP_CONFIG
import numpy as np

def filter_conditions(dataset: pd.DataFrame):
    print(dataset.columns)
    print(f"Before filtering: {dataset.shape[0]}")

    # Filter out Stocks... TODO: put this into filter interface and make configurable in model_config
    penny_stock_mask = (dataset["unadj_open"] >= 2)    
    # staleness_mask = (dataset["staleness"] < 1)  
    jaccard_mask = (dataset["jaccard"] < 0.9)  
    dollar_volume_mask = (dataset["dollar_volume"] >= 30_000) # illiquid stocks TODO: this has look-ahead bias (?)
    keywords = ["estimate", "dividend", "split"]
    keyword_mask = dataset["parsed_body"].apply(lambda x: any([k in x for k in keywords]))
    time_is_available = ~((dataset['news_time'].dt.hour == 0) & (dataset['news_time'].dt.minute == 0))
    
    mask_dict = dict(penny_stock_mask=penny_stock_mask, 
                    # staleness_mask=staleness_mask, 
                    jaccard_mask=jaccard_mask,
                    dollar_volume_mask=dollar_volume_mask,
                    keyword_mask=keyword_mask,
                    time_is_available=time_is_available,
                    )
    
    active_masks = [
        #'penny_stock_mask', 
        #'dollar_volume_mask', 
        'jaccard_mask', 
        'time_is_available']
    
    for name in mask_dict:
        print(f"{'(active) 'if name in active_masks else ''}{name}: {(~mask_dict[name]).sum()} entries affected")
    
    combined_mask = [all(column) for column in zip(*[mask_dict[mask] for mask in active_masks])]
    
    dataset = dataset.loc[combined_mask]
    
    # TODO: How many columns do we have? this might be too aggressive of a dropna
    print(dataset.columns)
    print("Dropping any NaNs...")
    dataset = dataset.dropna()
    
    print(f"After filtering: {dataset.shape[0]}")
    return dataset

def main():
    # ------------------------------------- Calculate target variables and additional features
    dat: pd.DataFrame = pd.read_parquet(path=config.data.merged)
    print(f"{dat.columns=}")
    
    # We need to filter before winsorizing, otherwise crazy outliers will affect max/min values a lot 
    dat = filter_conditions(dataset=dat)
    
    #! Adding indicators here... move this somewhere else at some point
    dat.loc[:, 'is_overnight_news'] == (dat.news['news_time'].dt.hour >= 16) | ((dat.news['news_time'].dt.hour <= 9) & (dat.news['news_time'].dt.minute < 30))
    
    # The larger the move in the overall market the less idiosyncratic the move in the stock will be
    # and hence will contain less information/ more noise w.r.t. to the company news.
    dat.loc[:, 'sample_weights'] = 1 / (1 + np.abs(dat['r_spy'])*100 ) ** 1.5
    
    dat.loc[:, "r_mkt_adj"] =  dat["r"] - dat["r_spy"]
    
    #TODO: This needs to be of r_mkt_adj, not of wahtever else std_252 is or?
    dat.loc[:, "z_score"] = dat["r_mkt_adj"]# / dat["std_252"]
    
    # Winsorizing
    dat.loc[:, "z_score"] = dat["z_score"].clip(lower=dat["z_score"].quantile(0.05), upper=dat["z_score"].quantile(0.95))
    # Scaling to avoid numerical issues
    dat.loc[:, "z_score"] = (dat.loc[:, "z_score"] / dat.loc[:, "z_score"].std())
    
    print(f"After filtering: {dat.shape[0]}")
    dat: pd.DataFrame = PREP_CONFIG.splitter.add_splits(dat)
    
    # Making balanced data set based on training data set
    train_dat = dat['split'] == 'training'
    upper_z_quantile = dat.loc[train_dat, "z_score"].quantile(0.666)
    lower_z_quantile = dat.loc[train_dat, "z_score"].quantile(0.333)
    median = dat.loc[train_dat, "z_score"].quantile(0.5)
   
    # Ordinal labeling
    dat.loc[:, "z_score_2_class"] = 0
    dat.loc[dat["z_score"] >= median, "z_score_2_class"] = 1
    print(dat["z_score_2_class"].value_counts())
    
    # Ordinal labeling
    dat.loc[:, "z_score_3_class"] = 0
    dat.loc[dat["z_score"] >= upper_z_quantile, "z_score_3_class"] = 1
    dat.loc[dat["z_score"] <= lower_z_quantile, "z_score_3_class"] = 2
    print(dat["z_score_3_class"].value_counts())
    
    dat.to_parquet(config.data.learning_dataset)

if __name__ == "__main__":
    print("Starting make_learning_dataset")
    
    dat = pd.read_parquet(config.data.learning_dataset)
    print(dat.describe())
    dat.describe().to_csv("data/learn_dat_describe.csv")
    
    main()
