import torch
from model.util import MyBertModel
from model.neural_net import validation_dataloader, test_dat
from model.util import predict
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
import plotly.io as pio
from evaluation import get_metrics
pio.renderers.default = "vscode"


# OBSOLOTE -> evaluation.ipynb

model = MyBertModel()

#Later to restore:
model.load_state_dict(torch.load("data/model"))
model.eval()

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")
model.to(device)


start = time.time()
y_pred_scaled = predict(model, validation_dataloader, device)


test_dat.loc[:, "Fcst"] = y_pred_scaled

end = time.time()
print(f"{start-end:.2f}s")


test_labels = test_dat.IntradayReturn.tolist()

print(f"Length of evaluation set: {len(y_pred_scaled)}")
print("Vanilla results:")
y_hat = y_pred_scaled
y = np.array(test_labels)
assert len(y_hat) == len(y)

mae, rw_mae, TP, TN = get_metrics(y_hat, y)
metrics_dict = dict(mae=[mae], mae_rw=[rw_mae], TP=[TP], TN=[TN])
metrics_df = pd.DataFrame.from_dict(metrics_dict)
print(metrics_df)


pred_margin_mask = np.abs(y_pred_scaled) >= 0.02

print(f"\nWith prediction margin mask:")
y_hat = y_pred_scaled[pred_margin_mask]
y = np.array(test_labels)[pred_margin_mask]
print(f"\nLength of prediction margin masked evaluation set: {len(y_hat)}")
mae, rw_mae, TP, TN = get_metrics(y_hat, y)
metrics_dict = dict(mae=[mae], mae_rw=[rw_mae], TP=[TP], TN=[TN])
metrics_df = pd.DataFrame.from_dict(metrics_dict)
print(metrics_df)


##############
# Import stocks
stocks = pd.read_pickle("data/stocks.pkl")
stocks.loc[:, "Date"] = pd.to_datetime(stocks.Date)
# TODO: Do same transformations as import in asset_data_preprocessor

# Analysis of single forecast: 
pr_time, ticker, fcst = test_dat.iloc[150, :][["Date", "ID", "Fcst"]]
df = stocks.query("(Date >= @pr_time) & (ID == @ticker)").head()
fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

fig.show()




