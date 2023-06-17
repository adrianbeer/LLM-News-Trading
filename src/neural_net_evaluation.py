import torch
from util import MyBertModel

model = MyBertModel()

#Later to restore:
model.load_state_dict(torch.load("data/model"))
model.eval()


# print(tokenizer.decode(input_ids[0].squeeze(), skip_special_tokens=True))
# print(len(input_ids[0]))

# Forecasting

# y_pred_scaled = predict(model, validation_dataloader, device)