# First execute asset_data_preprocessor (to get relevant list of tickers)
# Then news_importer
# Then news_parser (here tickers from assert_data_processor are used)
# Then data_merger, where price and news data are merged.
# Then neural_net.py, where the neural network is trained
# Then neural_net_evaluation where the performance of the neurlal network that was trained in the last step is eavluated

4. Authorenfeld für Daten von vor 2017 berichtigen und evtl. noch andere Felder? (Aktienbezug, Datum?)

5. can NN relate dividend and share price? JA -> Fundamentaldaten als Text einfügen.

7. Berücksichtigen wann die Nachrichten rauskommen und lieber C-C anstatt O-C betrachten, um, Gaps mitzuerfassen.

8. Was machen, wenn zwei nachrichten an einem Tag rauskommen? -> Löschen

9. Pharmaceuticals ausschliessen (Suche nach Phase 2b oder soetwas)

10. NN-Vorhersagen nutzen und Clustern, um besten Trades zu identifizieren.

------------------

# Langfristige Ideen:
- Sell-on-good-news
- Buy-on-bad-news
- GD-Richtungsfilter