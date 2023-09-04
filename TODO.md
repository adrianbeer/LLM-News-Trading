1. First execute asset_data_preprocessor (to get relevant list of tickers)
2. Then news_importer
3. Then news_parser (here tickers from assert_data_processor are used)
4. Then data_merger, where price and news data are merged.
5. Then neural_net.py, where the neural network is trained
6. Then neural_net_evaluation(.ipynb) where the performance of the neurlal network that was trained in the last step is eavluated

4. Authorenfeld für Daten von vor 2017 berichtigen und evtl. noch andere Felder? (Aktienbezug, Datum?)

5. can NN relate dividend and share price? JA -> Fundamentaldaten als Text einfügen.

7. Berücksichtigen wann die Nachrichten rauskommen und lieber C-C anstatt O-C betrachten, um, Gaps mitzuerfassen.

8. Was machen, wenn zwei nachrichten an einem Tag rauskommen? -> Löschen

9. Pharmaceuticals ausschliessen (Suche nach Phase 2b oder soetwas)

10. NN-Vorhersagen nutzen und Clustern, um besten Trades zu identifizieren.

11. Schätze Verteilungseigentschaften für die Kursbewegung nach Nachrichtne mit bestimmten Kennzeichen.
Z.B. wenn ein bestimmtes Wort vorkommt ist die Varianz/Erwartungswert größer etc.
------------------

# Langfristige Ideen:
- Sell-on-good-news
- Buy-on-bad-news
- GD-Richtungsfilter

11. Versuche unterschiedliche html-to-text-parser
newsText = ek.get_news_story(storyId) #get the news story
if newsText:
    soup = BeautifulSoup(newsText,"lxml") #create a BeautifulSoup object from our HTML news article
    sentA = TextBlob(soup.get_text())

12. Mass für Kursbewegung: GDRs/ADRs ?
