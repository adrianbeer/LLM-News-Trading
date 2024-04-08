
import pandas as pd
from src.config import config, MODEL_CONFIG
import numpy as np

def filter_conditions(dataset: pd.DataFrame):
    print(f"Before filtering: {dataset.shape[0]}")

    # Filter out Stocks... TODO: put this into filter interface and make configurable in model_config
    penny_stock_mask = (dataset["unadj_open"] >= 2)    
    staleness_mask = (dataset["staleness"] < 1)  
    dollar_volume_mask = (dataset["dollar_volume"] >= 30_000)
    keywords = ["estimate", "dividend", "split"]
    keyword_mask = dataset["parsed_body"].apply(lambda x: any([k in x for k in keywords]))
    
    
    mask_dict = dict(penny_stock_mask=penny_stock_mask, 
                    staleness_mask=staleness_mask, 
                    dollar_volume_mask=dollar_volume_mask,
                    keyword_mask=keyword_mask)
    
    for name in mask_dict:
        print(f"{name}: {(~mask_dict[name]).sum()} entries affected")
    
    dataset = dataset[
        penny_stock_mask &      # penny stocks
        #keyword_mask & 
        dollar_volume_mask &     # illiquid stocks TODO: this has look-ahead bias
        staleness_mask        # repeat news      
        ]
    
    # TODO: How many columns do we have? this might be too aggressive of a dropna
    dataset.dropna(inplace=True)
    return dataset

def main():
    # ------------------------------------- Calculate target variables and additional features
    dat: pd.DataFrame = pd.read_parquet(path=config.data.merged)
    print(f"{dat.columns=}")
    
    # We need to filter before winsorizing, otherwise crazy outliers will affect max/min values a lot 
    dat = filter_conditions(dataset=dat)
    
    # TODO: overnight news tag
    
    # The larger the move in the overall market the less idiosyncratic the move in the stock will be
    # and hence will contain less information/ more noise w.r.t. to the company news.
    dat.loc[:, 'sample_weights'] = 1 / (1 + np.abs(dat['r_spy'])*100 ) ** 1.5
    
    
    dat.loc[:, "r_mkt_adj"] =  dat["r"] - dat["r_spy"]
    
    #TODO: This needs to be of r_mkt_adj, not of wahtever else std_252 is or?
    dat.loc[:, "z_score"] = dat["r_mkt_adj"]# / dat["std_252"]
    
    # Winsorizing
    dat.loc[:, "z_score"] = dat["z_score"].clip(lower=dat["z_score"].quantile(0.05), upper=dat["z_score"].quantile(0.9))
    # Scaling to avoid numerical issues
    dat.loc[:, "z_score"] = (dat.loc[:, "z_score"] / dat.loc[:, "z_score"].std())
    
    # TODO: Calculate based on training set split
    upper_z_quantile = dat["z_score"].quantile(0.5)
    lower_z_quantile = dat["z_score"].quantile(0.0)
    
    # Ordinal labeling
    dat.loc[:, "z_score_class"] = 0
    dat.loc[dat["z_score"] >= upper_z_quantile, "z_score_class"] = 1
    #dat.loc[dat["z_score"] <= lower_z_quantile, "z_score_class"] = 0
    print(dat["z_score_class"].value_counts())

    print(f"After filtering: {dat.shape[0]}")

    dat: pd.DataFrame = MODEL_CONFIG.splitter.add_splits(dat)
    dat.to_parquet(config.data.learning_dataset)

if __name__ == "__main__":
    print("Starting make_learning_dataset")
    
    dat = pd.read_parquet(config.data.learning_dataset)
    print(dat.describe())
    dat.describe().to_csv("data/learn_dat_describe.csv")
    
    main()
