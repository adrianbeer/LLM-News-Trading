import numpy as np
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm

def parallelize_dataframe(df, func, n_cores=4):
    n_splits = n_cores*10
    df_split = np.array_split(df, n_splits)
    pool = Pool(n_cores)
    df = pd.concat(tqdm(pool.imap(func, df_split), total=n_splits))
    pool.close()
    pool.join()
    return df