import pandas as pd 
import random 

class Splitter:
    
    def __init__(self):
        pass
    
    def add_splits(self, dat: pd.DataFrame) -> pd.DataFrame:
        new_dat = self._split(dat)
        self.print_data_split_info(new_dat)
        return new_dat

    def _split(self, dat: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
    
    def print_data_split_info(self, dat: pd.DataFrame) -> None:
        train_N = dat[dat["split"] == "training"].shape[0]
        valid_N = dat[dat["split"] == "validation"].shape[0]
        test_N = dat[dat["split"] == "testing"].shape[0]
        print(f"{train_N} samples in training set."
            f"\n {valid_N} samples in validation set."
            f"\n {test_N} samples in testing set.")


class DateSplitter(Splitter):
    
    def __init__(self, val_cutoff_date, test_cutoff_date, time_column):
        self.val_cutoff_date = val_cutoff_date
        self.test_cutoff_date = test_cutoff_date
        self.time_column = time_column
        
    def _split(self, dat):
        dat["split"] = "training"
        dat.loc[dat[self.time_column] >= self.val_cutoff_date, "split"] = "validation"
        dat.loc[dat[self.time_column] >= self.test_cutoff_date, "split"] = "testing"
        dat["split"] = dat["split"].astype("category")
        return dat
    
    
class RatioSplitter(Splitter):
    
    def __init__(self, train_perc, val_perc):
        self.train_perc = train_perc
        self.val_perc = val_perc
    
    def _split(self, dat: pd.DataFrame) -> pd.DataFrame:
        N = dat.shape[0]
        dat["split"] = "training"
        
        # Shuffle rows to make selection random 
        dat = dat.sample(frac=1, random_state=42)
        dat.iloc[int(N * self.train_perc):, : ].loc[:, "split"] = "validation"
        dat.iloc[int(N * (self.train_perc + self.val_perc)):, : ].loc[:, "split"] = "testing"
        dat["split"] = dat["split"].astype("category")
        return dat