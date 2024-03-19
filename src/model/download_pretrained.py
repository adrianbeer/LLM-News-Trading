from transformers import BertTokenizerFast, BertModel, BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification

hf_id = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(hf_id)
tokenizer.save_pretrained(f"data/models/{hf_id}")

model = AutoModelForSequenceClassification.from_pretrained(hf_id)
model.save_pretrained(f"data/models/{hf_id}")


