from src.model.neural_network import BERTClassifier, initialize_final_layer_bias_with_class_weights
from src.model.data_loading import CustomDataModule
from lightning.pytorch.callbacks import StochasticWeightAveraging
import lightning as pl
from src.config import config, MODEL_CONFIG
import torch.nn as nn
import numpy as np
import pandas as pd

target_col_name = "z_score_class"

# Should be labeled with 0, 0, 1, 1, 2, 2
news_data_idx = [12460904, 12460977, 12460964, 12495579, 12460928, 12460928, 12495897]

def test_single_label_training_accuracy():
    model: nn.Module = BERTClassifier(bert_model_name=MODEL_CONFIG.base_model,
                                      num_classes=3,
                                      deactivate_bert_learning=True,
                                      learning_rate=0.01,
                                      class_weights=[0.3, 0.3, 0.4])
    weights = pd.Series(index=[0,1,2], data=[0.3, 0.3, 0.4])
    initialize_final_layer_bias_with_class_weights(model, weights)

    dm = CustomDataModule(news_data_path=config.data.learning_dataset, 
                          input_ids_path=config.data.news.input_ids, 
                          masks_path=config.data.news.masks, 
                          batch_size=len(news_data_idx),
                          target_col_name=target_col_name,
                          news_data_idx=news_data_idx)
    
    trainer = pl.Trainer(num_sanity_val_steps=2,
                     max_epochs=30,
                     gradient_clip_val=1,
                     callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
                     accumulate_grad_batches=1,
                     precision=16,
                     # Avoid logging in testing module
                     logger=False)
    
    trainer.fit(model, dm)
    logit_preds = trainer.predict(model, dm.train_dataloader())
    binary_preds = np.apply_along_axis(func1d=np.argmax, arr=logit_preds[0].numpy(), axis=1)
    assert (binary_preds == np.array([0, 0, 1, 1, 2, 2])).all()


def test_multi_label_training_accuracy():
    pass