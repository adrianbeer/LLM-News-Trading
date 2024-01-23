import numpy as np
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm

def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(tqdm(pool.imap(func, df_split), total=n_cores))
    pool.close()
    pool.join()
    return df