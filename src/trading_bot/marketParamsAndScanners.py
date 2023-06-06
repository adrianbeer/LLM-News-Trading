import ibapi
from ibapi.client import *
from ibapi.wrapper import *

import threading
import time

#import pydevd;pydevd.settrace(suspend=False)

# Class for IBKR connection
class TestApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def nextValidId(self, orderId: int):
        self.reqScannerParameters()

    def scannerParameters(self, xml):
        dir = "scanner.xml"
        open(dir, "w").write(xml)
        print("Scanner parameters received!")
        self.disconnect()

app = TestApp()
app.connect("127.0.0.1", 7496, 1001)
app.run()