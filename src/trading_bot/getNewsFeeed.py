import ibapi
from ibapi.client import *
from ibapi.common import ListOfNewsProviders
from ibapi.wrapper import *

import threading
import time

#import pydevd;pydevd.settrace(suspend=False)

# Class for IBKR connection
class IBApi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def contractDetails(self, reqId, contractDetails):
        print(f"contract details: {contractDetails.longName}, conId: {contractDetails.underConId}")


    def tickNews(self, tickerId: int, timeStamp: int, providerCode: str,
                    articleId: str, headline: str, extraData: str):
        print("TickNews. TickerId:", tickerId, 
              "\nTimestamp: ", timeStamp,
              "ProviderCode:", providerCode, "ArticleId:", articleId,
              "Headline:", headline, "ExtraData:", extraData)
        self.reqNewsArticle(10002, providerCode, articleId, [])
        

    def newsArticle(self, requestId: int, articleType: int, articleText: str):
        print(f"article_type: {articleType}, articleText: {articleText}")
        self.disconnect()


    def historicalNews(self, requestId: int, time: str, providerCode: str, articleId: str, headline: str):
            print("HistoricalNews. time:", time, 
              "ProviderCode:", providerCode, 
              "ArticleId:", articleId,
              "Headline:", headline)
            self.reqNewsArticle(10004, providerCode, articleId, [])
            #self.disconnect()

def main():
    # Start Bot
    app = IBApi()
    app.connect("127.0.0.1", 7496, 1)
    time.sleep(3) # run reqeust faster than socket connection can be build

    mycontract = Contract()
    mycontract.symbol = "AAPL"
    mycontract.secType = "STK"
    mycontract.exchange = "SMART"
    mycontract.currency = "USD"
    mycontract.primaryExchange = "ISLAND" # NASDAQ
    app.reqContractDetails(1, mycontract)

    contract = Contract()
    contract.symbol = "BZ:BZ_ALL"
    contract.secType = "NEWS"
    contract.exchange = "BZ"

    #app.reqMktData(1, contract, "mdoff,292", False, False, [])
    # 2023-05-01 00:00:00.0
    app.reqHistoricalNews(10003, 8314, "BZ", "", "", 300, [])

    app.run() # EWrapper run loop to receive requested data
    

if __name__ == "__main__":
    main()









