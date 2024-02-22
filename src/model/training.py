import warnings
from argparse import ArgumentParser

import lightning as pl
import torch
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.tuner import Tuner
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from src.config import MODEL_CONFIG, config
from src.model.bert_classifier import (
    BERTClassifier,
    initialize_final_layer_bias_with_class_weights,
)
from src.model.bert_regressor import BERTRegressor
from src.model.data_loading import CustomDataModule

warnings.filterwarnings("ignore", category=UserWarning)

pt_version = torch.__version__
print(f"[INFO] Current PyTorch version: {pt_version} (should be 2.x+)")
print(f'{torch.cuda.is_available()=}')
torch.set_float32_matmul_precision('high')


def get_model(ckpt, model_args) -> pl.LightningModule:
    if ckpt:
        model: pl.LightningModule = MODEL_CONFIG.neural_net.load_from_checkpoint(ckpt, 
                                                                                 **model_args)
        print(f"Using Checkpointed model at {ckpt}...")
    elif MODEL_CONFIG.task == "Regression":
        print("Initialize news regression model...")
        model: pl.LightningModule = BERTRegressor(bert_model_name=MODEL_CONFIG.pretrained_network,
                                                  **model_args)
    elif MODEL_CONFIG.task == "Classification":
        print("Initialize new Classification model...")
        dm.setup("fit")
        class_distribution = dm.get_class_distribution()
        print(dm.train_dataloader().dataset.get_class_distribution())
        model: pl.LightningModule = BERTClassifier(bert_model_name=MODEL_CONFIG.pretrained_network,
                                        num_classes=3,
                                        class_weights=1 / class_distribution.values,
                                        **model_args)
        initialize_final_layer_bias_with_class_weights(model, class_distribution)
    else:
        raise ValueError()
    return model

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--hidden_layer_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--deactivate_bert_learning", action='store_true')    
    parser.add_argument("--fast_dev_run", action='store_true')
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--stop_after_lr_finder", action='store_true', 
                        help="Creates data/lr_finder.png and stops afterwards in order to inspect learning_rates")
    
    args = parser.parse_args()

    model_args = dict(
        deactivate_bert_learning=args.deactivate_bert_learning,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        hidden_layer_size=args.hidden_layer_size,
    )

    dm = CustomDataModule(news_data_path=config.data.learning_dataset, 
                          input_ids_path=config.data.news.input_ids, 
                          masks_path=config.data.news.masks, 
                          batch_size=args.batch_size, # Batch size is configured automatically later on
                          target_col_name=MODEL_CONFIG.target_col_name)
    
    model: pl.LightningModule = get_model(args.ckpt, model_args)

    tb_logger = pl_loggers.TensorBoardLogger('tb_logs', 
                                             name="bert_regressor",
                                             flush_secs=120)
    
    #TODO change metrics
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    callbacks = [
        TuneReportCallback(metrics, on="validation_end"),
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=2, save_last=True),
        #StochasticWeightAveraging(swa_lrs=1e-2),
        ]
    trainer = pl.Trainer(num_sanity_val_steps=2,
                        max_epochs=20,
                        gradient_clip_val=1,
                        
                        callbacks=callbacks,
                        accumulate_grad_batches=5,
                        precision=16,
                        accelerator="gpu", 
                        devices=1,
                        logger=tb_logger,
                        fast_dev_run=args.fast_dev_run)
    tuner = Tuner(trainer)

    if args.batch_size is None:
        # Auto-scale batch size by growing it exponentially (default)
        tuner.scale_batch_size(model,
                               mode="power",
                               datamodule=dm)
        dm.batch_size = min(dm.batch_size,  512)

    if args.learning_rate is None:
        # Run learning rate finder
        lr_finder = tuner.lr_find(model, dm)
        fig = lr_finder.plot(suggest=True)
        fig.savefig('data/lr_finder.png')
        new_lr = lr_finder.suggestion()
        model.hparams.learning_rate = new_lr
        
    if args.stop_after_lr_finder:
        print("Created data/lr_finder.png and stopping.")
        exit

    print(f"Start training with {model.hparams.learning_rate=}")
    trainer.fit(model,
                dm,
                ckpt_path=args.ckpt)


