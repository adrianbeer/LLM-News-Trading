import os
import pandas as pd
from src.config import config 
from tqdm.auto import tqdm
tqdm.pandas()
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
from src.utils.dataframes import parallelize_dataframe, block_apply_factory
from src.preprocessing.news_preprocessing import body_formatter

'''
## Grobes HTML-Parsing
Als erstes müssen wir die HTML-Dokumente zu normalem Text umwandeln, 
ansonsten sind die Text-Zellen zu groß und führen zu Problemen mit PyArrow/Dask.
'''


# OOM after starting second loop... 
# something isn't being properly being garbage collected... maybe the child processes of multiprocessing in parallelize_dataframe?
for year in tqdm(range(2010, 2024)):
    print(f"{year}")
    df = pd.read_parquet(config.data.benzinga.raw + f"/story_df_raw_{year}.parquet")
    df["html_body"] = parallelize_dataframe(df["html_body"], 
                                            block_apply_factory(body_formatter), 
                                            n_cores=os.cpu_count())
    df = df.rename(columns={"html_body":"body"})
    df.to_parquet(config.data.benzinga.raw_html_parsed + \
                  f"/story_df_html_parsed{year}.parquet")