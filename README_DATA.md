# Location and Description of Data

GoogDrive:
=========
    data/ (chronological):
        `raw_bzg`: 
            - Raw news articles which were scraped using `src/preprocessing/news_downloader.py`
        `unraw1_bzg`: 
            - Converts HTML to plain text
        `unraw2_bzg`:
            - Just partitions the data set in a new way
        `unraw3_bzg`:
            - Remove news without ticker information
            - Extract the news source/author
            - Only keep news from "PRNewswire", "Globe Newswire", "Business Wire" and "ACCESSWIRE"
        `latest`: 
            - Get full company name from ticker
            - Remove tickers for which yahoo doesn't return a company name
            --> Create `ticker_name_mapper.parquet`
        `latest2`/`latest1`:
            - Remove tickers for which the company name doesn't appear in the text
            - Remove duplicate entries
        `parsed_bzg`:
            - Fully parsed news messages, ready to be tokenized

data_shared/
    `ticker_name_mapper.parquet`:
        - Maps company tickers to company name


Google Cloud Storage:
======================
    `gcs://extreme-lore-398917-bzg/latest2/`
        - same as `latest2` from GoogDrive


Intenso-2TB Hard Drive:
===================
    `IQFeedData`
        - 1min stock prices from IQFeed in parquet data format (one file for each stock).


USB-K:
======
    Downloaded news from Benzinga in raw format, see `src/preprocessing/news_downloader.py`.
