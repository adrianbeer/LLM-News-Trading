
import pandas as pd
from src.config import config, MODEL_CONFIG

def main():
    # ------------------------------------- Calculate target variables and additional features
    dat: pd.DataFrame = pd.read_parquet(path=config.data.merged)

    # TODO: overnight news tag
    dat.loc[:, "r_mkt_adj"] =  dat["r"] - dat["r_spy"]
    #TODO: This needs to be of r_mkt_adj, not of wahtever else std_252 is or?
    dat.loc[:, "z_score"] = dat["r_mkt_adj"] / dat["std_252"]
    dat.loc[:, "z_score"] = dat["z_score"].clip(lower=dat["z_score"].quantile(0.05), upper=dat["z_score"].quantile(0.95))

    # TODO: Calculate based on training set split
    upper_z_quantile = 0.27
    lower_z_quantile = -0.27
    dat.loc[:, "z_score_class"] = 1
    # Ordinal labeling
    dat.loc[dat["z_score"] >= upper_z_quantile, "z_score_class"] = 2
    dat.loc[dat["z_score"] <= lower_z_quantile, "z_score_class"] = 0
    dat["z_score_class"].value_counts()
    dat.to_parquet(path=config.data.merged)


    # ------------Final filtering of data and make learning_dataset
    # Download dataset
    dataset = pd.read_parquet(config.data.merged)
    print(f"Before filtering: {dataset.shape[0]}")

    # Filter out Stocks... TODO: put this into filter interface and make configurable in model_config
    dataset = dataset[
        (dataset["unadj_open"] >= 1) &         # penny stocks
        (dataset["dollar_volume"] >= 30_000) & # illiquid stocks TODO: this has look-ahead bias
        (dataset["staleness"] <= 0.95)          # repeat news      
        ]
    # How many columns do we have? this might be too aggressive of a dropna
    dataset.dropna(inplace=True)

    print(f"After filtering: {dataset.shape[0]}")

    dataset: pd.DataFrame = MODEL_CONFIG.splitter.add_splits(dataset)
    dataset.to_parquet(config.data.learning_dataset)

if __name__ == "__main__":
    main()
