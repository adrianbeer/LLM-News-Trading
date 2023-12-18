# PROJECT B U G A T T I

## Data
Description for the files and processing steps can be found in `README_DATA.md`.

## Warnings
- Pylance breaks everything.

## Instruktionen
1. First execute asset_data_preprocessor (to get relevant list of tickers)
2. Then news_importer
3. Then news_parser (here tickers from assert_data_processor are used)
4. Then data_merger, where price and news data are merged.
5. Then neural_net.py, where the neural network is trained
6. Then neural_net_evaluation(.ipynb) where the performance of the neurlal network that was trained in the last step is eavluated

