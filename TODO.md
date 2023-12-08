--------------------------------
Big Picture:
1. Tai-Pan Daten fetchen (Oktober), säubern und zusammenführen und input-output-Paare erstellen.

2. Schauen ob IQFeed über R-Bibliothek leicht zu benutzen ist. Ansonsten Kallsen fragen, ob Uni Kauff der kibot.com-Daten übernehmen würde.

3. Ami-Daten bestellen und zusammenführen.

4. Cluster based on CLS token

5. Topics sollten als scalar nicht als one-hot-variable übergeben werden, um mehr informationen in derselben variable mitgeben zu können.

--------------------------------

4. Sobald wir ein Mapping von Nachrichten zu Kursen haben, können wir versuchen einen KNN Algorithmus basierend auf
der Word Mover's Distance zu benutzen, bzw. generell Clustering-Algorithmen mit diesem Distanzmass benutzen, siehe
[Gensim](https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html#sphx-glr-auto-examples-tutorials-run-wmd-py).


5. can NN relate dividend and share price? JA -> Fundamentaldaten als Text einfügen.

7. Berücksichtigen wann die Nachrichten rauskommen und lieber C-C anstatt O-C betrachten, um, Gaps mitzuerfassen.

8. Was machen, wenn zwei nachrichten an einem Tag rauskommen? -> Löschen

9. Pharmaceuticals ausschliessen (Suche nach Phase 2b oder soetwas)

10. NN-Vorhersagen nutzen und Clustern, um besten Trades zu identifizieren.


Dateparser -> "XXX days ago" anstatt "previous date" ? 
-> XXX days in the futured anstatt "a future date" ? 

11. Schätze Verteilungseigentschaften für die Kursbewegung nach Nachrichtne mit bestimmten Kennzeichen.
Z.B. wenn ein bestimmtes Wort vorkommt ist die Varianz/Erwartungswert größer etc.
------------------

# Topicextraktion:
- Können Text2Topic benutzen zum beispiel...
- Oder ChatGPT

# Langfristige Ideen:
- Sell-on-good-news
- Buy-on-bad-news
- GD-Richtungsfilter
- Dividendenerhöhuung

11. Versuche unterschiedliche html-to-text-parser
newsText = ek.get_news_story(storyId) #get the news story
if newsText:
    soup = BeautifulSoup(newsText,"lxml") #create a BeautifulSoup object from our HTML news article
    sentA = TextBlob(soup.get_text())

12. Mass für Kursbewegung: GDRs/ADRs ?

