
# Big Picture/ Long-Term:

-  Tai-Pan Daten fetchen (Oktober), säubern und zusammenführen und input-output-Paare erstellen.

-  Ami-Daten bestellen und zusammenführen.

-  Cluster based on CLS token

-  Topics sollten als scalar nicht als one-hot-variable übergeben werden, um mehr informationen in derselben variable mitgeben zu können.

- Bootstrap trade matrix

# Short-Term:
- news parser -> about company block wegbekommen... und firstcall...

- Pipeline everything!!

- Stratifizierung nach Unternehmen oder so? Google hat viel mehr Nachrichten als z.b. ein kleineres Unternehmen.

-  Sobald wir ein Mapping von Nachrichten zu Kursen haben, können wir versuchen einen KNN Algorithmus basierend auf
der Word Mover's Distance zu benutzen, bzw. generell Clustering-Algorithmen mit diesem Distanzmass benutzen, siehe
[Gensim](https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html#sphx-glr-auto-examples-tutorials-run-wmd-py).

-  can NN relate dividend and share price? JA -> Fundamentaldaten als Text einfügen.

- Was machen, wenn zwei nachrichten an einem Tag rauskommen? -> Löschen

-  Pharmaceuticals ausschliessen (Suche nach Phase 2b oder soetwas) -> Warum??

- Dateparser -> "XXX days ago" anstatt "previous date" ? 
  -> XXX days in the futured anstatt "a future date" ? 

1.  Schätze Verteilungseigentschaften für die Kursbewegung nach Nachrichtne mit bestimmten Kennzeichen.
Z.B. wenn ein bestimmtes Wort vorkommt ist die Varianz/Erwartungswert größer etc.


# Specific Topics::

## Evaluations-Ideen:
- Sell-on-good-news
- Buy-on-bad-news
- GD-Richtungsfilter
- Dividendenerhöhuung
- Overnight vs Intra-day News
- Benutze Volumen als Proxy für Marketkapitalisierung

## Topicextraktion:
- Können Text2Topic benutzen zum beispiel...
- News2vec: News Network Embedding with Subnode Information
- Oder ChatGPT

- Versuche unterschiedliche html-to-text-parser
newsText = ek.get_news_story(storyId) #get the news story
if newsText:
    soup = BeautifulSoup(newsText,"lxml") #create a BeautifulSoup object from our HTML news article
    sentA = TextBlob(soup.get_text())

- Mass für Kursbewegung: GDRs/ADRs ?


