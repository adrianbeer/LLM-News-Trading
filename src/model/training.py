import torch
import torch.nn as nn
from src.config import config, MODEL_CONFIG
import lightning as pl
from lightning.pytorch import loggers as pl_loggers

from src.model.bert_classifier import BERTClassifier, initialize_final_layer_bias_with_class_weights
from src.model.bert_regressor import BERTRegressor
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import StochasticWeightAveraging, ModelCheckpoint, LearningRateMonitor
from src.model.data_loading import CustomDataModule
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

pt_version = torch.__version__
print(f"[INFO] Current PyTorch version: {pt_version} (should be 2.x+)")
print(f'{torch.cuda.is_available()=}')
torch.set_float32_matmul_precision('high')

# Settings
automatic_learning_rate = False
learning_rate = 1e-6

automatic_batch_size = True
batch_size = 2

deactivate_bert_learning = False
ckpt = None
# ckpt = "tb_logs/bert_regressor/version_5/checkpoints/epoch=9-step=1704.ckpt"
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
                                            class_weights=1 / class_distribution.values)
    initialize_final_layer_bias_with_class_weights(model, class_distribution)
    return model


if __name__ == "__main__":
    dm = CustomDataModule(news_data_path=config.data.learning_dataset, 
                        input_ids_path=config.data.benzinga.input_ids, 
                        masks_path=config.data.benzinga.masks, 
                        batch_size=batch_size, # Batch size is configured automatically later on
                        target_col_name=MODEL_CONFIG.target_col_name)
    
    
    
    if ckpt:
        model: pl.LightningModule = MODEL_CONFIG.neural_net.load_from_checkpoint(ckpt, 
                                                                                 deactivate_bert_learning=deactivate_bert_learning,
                                                                                 learning_rate=learning_rate)
    
    elif MODEL_CONFIG.task == "Regression":
        model: pl.LightningModule = initialize_regressor()
    
    elif MODEL_CONFIG.task == "Classification":
        dm.setup("fit")
        class_distribution = dm.get_class_distribution()
        print(dm.train_dataloader().dataset.get_class_distribution())
        model = initialize_classifier(class_distribution)
    else:
        raise ValueError()

    tb_logger = pl_loggers.TensorBoardLogger('tb_logs', 
                                             name="bert_regressor",
                                             flush_secs=120)
    trainer = pl.Trainer(num_sanity_val_steps=2,
                        max_epochs=20,
                        gradient_clip_val=1,
                        #StochasticWeightAveraging(swa_lrs=1e-2),
                        callbacks=[
                            LearningRateMonitor(logging_interval='step'),
                            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=2, save_last=True),
                            ],
                        accumulate_grad_batches=5,
                        precision=16,
                        accelerator="gpu", 
                        devices=1,
                        logger=tb_logger)
    tuner = Tuner(trainer)

    if automatic_batch_size:
        # Auto-scale batch size by growing it exponentially (default)
        tuner.scale_batch_size(model,
                               mode="power",
                               datamodule=dm)
        dm.batch_size = min(dm.batch_size,  512)

    if automatic_learning_rate:
        # Run learning rate finder
        lr_finder = tuner.lr_find(model, dm)
        fig = lr_finder.plot(suggest=True)
        fig.savefig('data/lr_finder.png')
        new_lr = lr_finder.suggestion()
        model.hparams.learning_rate = new_lr

    print(f"Start training with {model.hparams.learning_rate=}")
    trainer.fit(model,
                dm,
                ckpt_path=ckpt)


