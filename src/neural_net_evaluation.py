import torch
from util import MyBertModel
from neural_net import validation_dataloader
from util import predict

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

# print(tokenizer.decode(input_ids[0].squeeze(), skip_special_tokens=True))
# print(len(input_ids[0]))

# Forecasting

y_pred_scaled = predict(model, validation_dataloader, device)

print("End")