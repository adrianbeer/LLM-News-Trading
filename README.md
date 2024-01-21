# PROJECT B U G A T T I

## Data
Description for the files and processing steps can be found in `README_DATA.md`.

## Warnings
- Pylance breaks everything.


## Interesting Links:
- https://stackoverflow.com/questions/25665114/understanding-interactive-brokers-tick-events
- https://epchan.blogspot.com/2015/04/beware-of-low-frequency-data.html


## Instruktionen
1. First execute asset_data_preprocessor (to get relevant list of tickers)
2. Then news_importer
3. Then news_parser (here tickers from assert_data_processor are used)
4. Then data_merger, where price and news data are merged.
5. Then neural_net.py, where the neural network is trained
6. Then neural_net_evaluation(.ipynb) where the performance of the neurlal network that was trained in the last step is eavluated

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