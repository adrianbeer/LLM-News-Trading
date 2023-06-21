import torch
from util import MyBertModel
from neural_net import validation_dataloader, test_dat
from util import predict
import numpy as np
import pandas as pd
import time

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
end = time.time()
print(f"{start-end:.2f}s")


def get_metrics(y, y_hat):
    mae = np.abs(y_hat - y).mean()
    rw_mae =  (np.abs(y)).mean()
    TP = ((y_hat > 0)  & (y > 0)).mean()
    TN = ((y_hat < 0)  & (y < 0)).mean()
    return mae, rw_mae, TP, TN

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


pred_margin_mask = np.abs(y_pred_scaled) >= 0.03

print(f"\nWith prediction margin mask:")
y_hat = y_pred_scaled[pred_margin_mask]
y = np.array(test_labels)[pred_margin_mask]
print(f"\nLength of prediction margin masked evaluation set: {len(y_hat)}")
mae, rw_mae, TP, TN = get_metrics(y_hat, y)
metrics_dict = dict(mae=[mae], mae_rw=[rw_mae], TP=[TP], TN=[TN])
metrics_df = pd.DataFrame.from_dict(metrics_dict)
print(metrics_df)
