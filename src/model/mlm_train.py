from transformers import RobertaConfig, RobertaModel, RobertaForMaskedLM, RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from src.model.data_loading import MLMDataset

configuration = RobertaConfig(vocab_size = 30000,
                              hidden_size = 256,
                              num_hidden_layers = 6,
                              num_attention_heads = 4,
                              intermediate_size = 1556,
                              hidden_act = 'gelu',
                              hidden_dropout_prob = 0.1,
                              attention_probs_dropout_prob = 0.1,
                              max_position_embeddings = 258,
                              type_vocab_size = 2,
                              initializer_range = 0.02,
                              layer_norm_eps = 1e-12,
                              pad_token_id = 1,
                              bos_token_id = 0,
                              eos_token_id = 2,
                              position_embedding_type = 'absolute',
                              classifier_dropout = None)


model = RobertaModel(configuration)
print(model.num_parameters())
model.save_pretrained("data/models/roberta_base")


model = RobertaForMaskedLM(config=configuration)
tokenizer = RobertaTokenizerFast.from_pretrained("data/models/newstokenizer", max_len=256)

# Define the Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="data/models/roberta_mlm",
    logging_dir='tb_logs/mlm',
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    adam_epsilon=1e-6,
    max_steps=500000,
    warmup_steps=10000,
    weight_decay=0.01,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    save_total_limit=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=MLMDataset(evaluate=False),
    eval_dataset=MLMDataset(evaluate=True),
    data_collator=data_collator,
)

trainer.train()