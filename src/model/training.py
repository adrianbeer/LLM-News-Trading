import torch
import torch.nn as nn
from transformers import BertTokenizerFast
from src.config import config, MODEL_CONFIG
import lightning as pl
from lightning.pytorch import loggers as pl_loggers

from src.model.bert_classifier import BERTClassifier
from src.model.bert_regressor import BERTRegressor
from src.model.neural_network import initialize_final_layer_bias_with_class_weights
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import StochasticWeightAveraging, ModelCheckpoint
from src.model.data_loading import CustomDataModule
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

pt_version = torch.__version__
print(f"[INFO] Current PyTorch version: {pt_version} (should be 2.x+)")
print(f'{torch.cuda.is_available()=}')


# Settings
tokenizer = BertTokenizerFast.from_pretrained(MODEL_CONFIG.tokenizer)
learning_rate = 1e-3 # or "automatic"
batch_size = 256 # or "automatic"
deactivate_bert_learning = False
ckpt = None
# ckpt = "C:/Users/Adria/Documents/Github Projects/trading_bot/lightning_logs/version_19/checkpoints/epoch=9-step=346.ckpt"


def initialize_regressor():
    model: pl.LightningModule = BERTRegressor(bert_model_name=MODEL_CONFIG.pretrained_network,
                                            deactivate_bert_learning=deactivate_bert_learning,
                                            learning_rate=learning_rate)
    return model


def initialize_classifier(class_distribution):
    model: pl.LightningModule = BERTClassifier(bert_model_name=MODEL_CONFIG.pretrained_network,
                                    num_classes=3,
                                    deactivate_bert_learning=deactivate_bert_learning,
                                    learning_rate=learning_rate,
                                    class_weights=class_weights)
    initialize_final_layer_bias_with_class_weights(model, class_distribution)
    return model


if __name__ == "__main__":
    dm = CustomDataModule(news_data_path=config.data.learning_dataset, 
                        input_ids_path=config.data.benzinga.input_ids, 
                        masks_path=config.data.benzinga.masks, 
                        batch_size=batch_size, # Batch size is configured automatically later on
                        target_col_name=MODEL_CONFIG.target_col_name)
    dm.setup("fit")
    
    if ckpt:
        model: pl.LightningModule = MODEL_CONFIG.neural_net.load_from_checkpoint(ckpt, deactivate_bert_learning=False)
        
    if MODEL_CONFIG.neural_net.task == "Regression":
        model: pl.LightningModule = initialize_regressor()
        print(f"baseline MAE: {dm.get_baseline_mae()}")
        
    if MODEL_CONFIG.neural_net.task == "Classification":    
        class_distribution = dm.get_class_distribution()
        class_weights = 1 / class_distribution.values
        print(dm.get_class_distribution())
        model = initialize_classifier(class_distribution)

    # tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs")
    model_checkpoint = ModelCheckpoint(monitor="val_loss",
                                       mode="min", 
                                       save_top_k=2)

    trainer = pl.Trainer(num_sanity_val_steps=2,
                        max_epochs=10,
                        gradient_clip_val=1,
                        callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
                        accumulate_grad_batches=10,
                        precision=16,
                        accelerator="gpu", 
                        devices=1,
                        model_checkpoint=model_checkpoint,)
    tuner = Tuner(trainer)

    if batch_size == "automatic":
        # Auto-scale batch size by growing it exponentially (default)
        tuner.scale_batch_size(model, 
                            mode="power", 
                            datamodule=dm)

    if learning_rate == "automatic":
        # Run learning rate finder
        lr_finder = tuner.lr_find(model, dm)
        fig = lr_finder.plot(suggest=True)
        fig.savefig('data/lr_finder.png')
        new_lr = lr_finder.suggestion()
        print(f"{new_lr=}")
        model.hparams.learning_rate = new_lr

    trainer.fit(model,
                dm,
                ckpt_path=ckpt)


