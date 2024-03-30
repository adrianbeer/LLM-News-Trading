
import pandas as pd
from src.config import config, MODEL_CONFIG
import numpy as np

def main():
    # ------------------------------------- Calculate target variables and additional features
    dat: pd.DataFrame = pd.read_parquet(path=config.data.merged)
    print(f"{dat.columns=}")
    
    # TODO: overnight news tag
    
    # The larger the move in the overall market the less idiosyncratic the move in the stock will be
    # and hence will contain less information/ more noise w.r.t. to the company news.
    dat.loc[:, 'sample_weights'] = 1 / (1+np.abs(dat['r_spy'])*100)**1.5
    
    
    dat.loc[:, "r_mkt_adj"] =  dat["r"] - dat["r_spy"]
    
    #TODO: This needs to be of r_mkt_adj, not of wahtever else std_252 is or?
    dat.loc[:, "z_score"] = dat["r_mkt_adj"] / dat["std_252"]
    
    # Winsorizing
    dat.loc[:, "z_score"] = dat["z_score"].clip(lower=dat["z_score"].quantile(0.05), upper=dat["z_score"].quantile(0.95))


    # TODO: Calculate based on training set split
    upper_z_quantile = dat["z_score"].quantile(0.75)
    lower_z_quantile = dat["z_score"].quantile(0.25)
    
    # Ordinal labeling
    dat.loc[:, "z_score_class"] = 1
    dat.loc[dat["z_score"] >= upper_z_quantile, "z_score_class"] = 2
    dat.loc[dat["z_score"] <= lower_z_quantile, "z_score_class"] = 0
    dat["z_score_class"].value_counts()
    
    dat.to_parquet(path=config.data.merged)

    # ------------Final filtering of data and make learning_dataset
    # Download dataset
    dataset = pd.read_parquet(config.data.merged)
    print(f"Before filtering: {dataset.shape[0]}")

    # Filter out Stocks... TODO: put this into filter interface and make configurable in model_config
    penny_stock_mask = (dataset["unadj_open"] >= 1)    
    staleness_mask = (dataset["staleness"] < 1)  
    dollar_volume_mask = (dataset["dollar_volume"] >= 30_000)
    keywords = ["beats", "estimates", "dividend", "increase", "split", "expected"]
    keyword_mask = dataset["parsed_body"].apply(lambda x: any([k in x for k in keywords]))
    
    
    mask_dict = dict(penny_stock_mask=penny_stock_mask, 
                    staleness_mask=staleness_mask, 
                    dollar_volume_mask=dollar_volume_mask,
                    keyword_mask=keyword_mask)
    
    for name in mask_dict:
        print(f"{name}: {(~mask_dict[name]).sum()} entries affected")
    
    dataset = dataset[
        penny_stock_mask &      # penny stocks
        keyword_mask & 
        dollar_volume_mask #&     # illiquid stocks TODO: this has look-ahead bias
        #staleness_mask        # repeat news      
        ]
    
    # TODO: How many columns do we have? this might be too aggressive of a dropna
    dataset.dropna(inplace=True)

    print(f"After filtering: {dataset.shape[0]}")

    dataset: pd.DataFrame = MODEL_CONFIG.splitter.add_splits(dataset)
    dataset.to_parquet(config.data.learning_dataset)

if __name__ == "__main__":
    print("Starting make_learning_dataset")
    main()
