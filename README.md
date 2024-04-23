# PROJECT B U G A T T I

## Interesting Links:
- https://stackoverflow.com/questions/25665114/understanding-interactive-brokers-tick-events
- https://epchan.blogspot.com/2015/04/beware-of-low-frequency-data.html

# Instructions 
## 1. Downloading
Download data using the modules in src.downloaders etc.

## 2. Preprocessing Pipeline
ticker_name_mapping requires an internet connection!!
1. python -m src.preprocessing.news_preprocessing --ticker_name_mapping
1. python -m src.preprocessing.news_preprocessing --process_body

1. <!-- python -m src.preprocessing.staleness --generate_cls_tokens --batchsize <batchsize> -->
1. <!-- python -m src.preprocessing.staleness --calculate_staleness -->
1. python -m src.preprocessing.data_merger initial_merge
1. python -m src.preprocessing.data_merger merge_overnight_news
1. python -m src.preprocessing.data_merger merge_daily_indicators

IF anything stripper should be applied AFTER initial merge... otherwise we would have to merge again... 
1. python -m src.preprocessing.news_preprocessing --stripper

1. python -m src.preprocessing.adding_jaccard
1. python -m src.preprocessing.make_learning_dataset

#! MLM Trainer should have access only to `parsed_body` (mayber `merged_msg` in future?) of the **training** set.

# both of these require the GPU!
1. python -m src.preprocessing.tokenizer
1. python -m src.model.mlm_train

## 3. Training
See `python -m src.model.training -h` for more info.

## 4. Predictions:
python -m src.evaluation.predictions

# Miscellaneous 
## Google Colab
Keep alive by pasting the following in the the developer console:
```
function ClickConnect(){
    console.log("Working");
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect,60000)
```

Add to the start of notebook for correct configuration
```
google_colab = True
if google_colab:
    from google.colab import drive
    import os
    drive.mount('/content/drive')
    cwd="/content/drive/MyDrive/NewsTrading/trading_bot"
    %cd /content/drive/MyDrive/NewsTrading/trading_bot
    %pip install -r requirements_clean.txt
    os.environ["TRADING_BOT_CONFIG_PATH"] = "src/config_gcs.yaml"
```