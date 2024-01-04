import numpy as np
from sklearn.metrics import accuracy_score

METRICS_FUNCTION_DICT = {
    "MAE": (lambda y_hat, y: np.abs(y_hat - y).mean()),
    "RW-MAE": (lambda y_hat, y: (np.abs(y)).mean()),
    "Accuracy": (lambda y_hat, y: accuracy_score((y > 0), (y_hat > 0)))
}

def get_metrics(y, y_hat):
    mae = np.abs(y_hat - y).mean()
    rw_mae =  (np.abs(y)).mean()
    TP = ((y_hat > 0)  & (y > 0)).mean()
    TN = ((y_hat < 0)  & (y < 0)).mean()
    return mae, rw_mae, TP, TN