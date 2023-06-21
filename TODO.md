# First execute asset_data_preprocessor (to get relevant list of tickers)
# Then news_importer
# Then news_parser (here tickers from assert_data_processor are used)
# Then data_merger, where price and news data are merged.
# Then neural_net.py, where the neural network is trained
# Then neural_net_evaluation where the performance of the neurlal network that was trained in the last step is eavluated




1. Ermögliche es Bert mit den gelernten Parametern in data/model zu initialisieren.

2. Neue Gewichte vom Training auch im Tokenizer berücksichtigen?

3. Train-test split -> Auslagern

4. re.sub zu bündeln ist wahrscheinlich schneller/effizienter als mehrere re.sub calls zu machen

5. can NN relate dividend and share price?

6. Intradayreturn targete variable kann nur benutzt werden, wenn auf nachrichten gefiltert wird, die vor börsenschluss herauskamen