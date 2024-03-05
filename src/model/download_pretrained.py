from transformers import BertTokenizerFast, BertModel

model = BertTokenizerFast.from_pretrained("yiyanghkust/finbert-pretrain")
model.save_pretrained("data/models/tokenizers/finbert_pretrain")

model = BertModel.from_pretrained("yiyanghkust/finbert-pretrain")
model.save_pretrained("data/models/networks/finbert_pretrain")