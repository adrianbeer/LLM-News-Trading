import ibapi
from ibapi.client import *
from ibapi.wrapper import *

import threading
import time

#import pydevd;pydevd.settrace(suspend=False)

# Class for IBKR connection
class IBApi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def contractDetails(self, reqId, contractDetails):
        print(f"contract details: {contractDetails.longName}")

    def contractDetailsEnd(self, reqId):
        # But safe place to disconnect
        print("End of contractDetails")
        self.disconnect()


def main():
    # Start Bot
    app = IBApi()
    app.connect("127.0.0.1", 7496, 1)
    
    mycontract = Contract()
    mycontract.symbol = "AAPL"
    mycontract.secType = "STK"
    mycontract.exchange = "SMART"
    mycontract.currency = "USD"
    mycontract.primaryExchange = "ISLAND" # NASDAQ

    time.sleep(3) # run reqeust faster than socket connectiono can be build

    app.reqContractDetails(1, mycontract)
    
    app.run() # EWrapper run loop to receive requested data
    

if __name__ == "__main__":
    main()



# class Bot:
#     def __init__(self):
#         self.ib = IBApi()
#         self.ib.connect("127.0.0.1", 7496, 1)

#         ib_thread = threading.Thread(target=self.run_loop, daemon=True)
#         ib_thread.start()

#         time.sleep(2)

#         print(self.ib.reqNewsProviders())
#         print(self.ib.newsProviders(self.ib.reqNewsProviders()))
#         self.ib.disconnect()

#     def run_loop(self):
#         self.ib.run()









