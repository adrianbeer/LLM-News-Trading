import ibapi
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

import threading
import time

#import pydevd;pydevd.settrace(suspend=False)

# Class for IBKR connection
class IBApi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)



class Bot:
    
    def __init__(self):
        self.ib = IBApi()
        self.ib.connect("127.0.0.1", 7496, 1)

        ib_thread = threading.Thread(target=self.run_loop, daemon=True)
        ib_thread.start()

        time.sleep(2)

        self.ib.newsProviders(self.ib.reqNewsProviders())
        print(self.ib.newsProviders(self.ib.reqNewsProviders()))
        self.ib.disconnect()

    def run_loop(self):
        self.ib.run()

# Start Bot
bot = Bot()










