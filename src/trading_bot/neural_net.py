from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch


text = "Hey"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Truncation = True as bert can only take inputs of max 512 tokens.
# return_tensors = "pt" makes the funciton return PyTorch tensors
# tokenizer.encode_plus specifically returns a dictionary of values instead of just a list of values
encoding = tokenizer.encode_plus(text, add_special_tokens = True, truncation = True, padding = "max_length", return_attention_mask = True, return_tensors = "pt")


from transformers import BertModel
class Bert_Model(nn.Module):
   def __init__(self):
       super(Bert_Model, self).__init__()
       self.bert = BertModel.from_pretrained('bert-base-uncased')
       D_in, D_out = self.bert.config.hidden_size, 1
       self.out = nn.Linear(D_in, D_out)
   def forward(self, input):
       _, output = self.bert(**input)
       out = self.out(output)
       return out

model = BertModel()

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")
model.to(device)

# Forward pass
# output = model(**encoding) 

# Optimizer, scheduler and loss function
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)

epochs = 5
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,       
                 num_warmup_steps=0, num_training_steps=total_steps)
loss_function = nn.MSELoss()