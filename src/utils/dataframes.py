import numpy as np
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm


def block_apply_factory(func, axis=None):
    global _f
    def _f(s):
        # s can be series or dataframe
        print(' ', end='', flush=True)
        ret = s.progress_apply(func, axis=axis) if axis else s.progress_apply(func)
        return ret
    return _f
    

def parallelize_dataframe(df, func, n_cores=4):
    print(f"{n_cores=}, {df.shape=}")
    n_splits = n_cores
    df_split = np.array_split(df, n_splits)
    pool = Pool(n_cores)
    try:
        df = pd.concat(tqdm(pool.imap(func, df_split), total=n_splits, desc="parallelize_data"))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    except Exception as e:
        print(e)
    pool.close()
    pool.join()
    return df