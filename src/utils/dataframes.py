import numpy as np
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm


def block_apply_factory(func):
    global _f
    def _f(s: pd.Series):
        print(' ', end='', flush=True)
        return s.progress_apply(func)
    return _f
    

def parallelize_dataframe(df, func, n_cores=4):
    print(f"{n_cores=}, {df.shape=}")
    n_splits = n_cores
    df_split = np.array_split(df, n_splits)
    pool = Pool(n_cores)
    df = pd.concat(tqdm(pool.imap(func, df_split), total=n_splits, desc="parallelize_data"))
    pool.close()
    pool.join()
    return df