import sys
import warnings
from argparse import ArgumentParser

import lightning as pl
import torch
import wandb
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

from src.config import MODEL_CONFIG
from src.config import config as DATA_CONFIG
from src.model.data_loading import CustomDataModule
from src.model.neural_network import get_model

warnings.filterwarnings("ignore", category=UserWarning)

pt_version = torch.__version__
print(f"[INFO] Current PyTorch version: {pt_version} (should be 2.x+)")
print(f'{torch.cuda.is_available()=}')
torch.set_float32_matmul_precision('high')

WANDB_PROJECT = 'news_trading'

def train_func(config: dict = None):
    run = wandb.init()
    if config is None: 
        config = wandb.config
    else: 
        wandb.config.update(config)
        
    print(f"{run.settings.mode=}")
    print(config)
    
    model_args = dict((k, config[k]) for k in ('deactivate_bert_learning', 
                                               'learning_rate', 
                                               'dropout_rate', 
                                               'hidden_layer_size'))

    dm = CustomDataModule(news_data_path=DATA_CONFIG.data.learning_dataset, 
                          input_ids_path=DATA_CONFIG.data.news.input_ids, 
                          masks_path=DATA_CONFIG.data.news.masks, 
                          batch_size=config["batch_size"],
                          target_col_name=MODEL_CONFIG.target_col_name)
    
    model: pl.LightningModule = get_model(config.get('ckpt'), model_args, dm)

    wandb_logger = WandbLogger(log_model=False, 
                               group='group',
                               offline=True)
    # tensor_logger = TensorBoardLogger("tb_logs", 
    #                                   name="my_model")

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=2, save_last=True),
        #StochasticWeightAveraging(swa_lrs=1e-2),
        #TriggerWandbSyncLightningCallback()
        ]
    trainer = pl.Trainer(
        # num_sanity_val_steps=2,
        max_epochs=config["epochs"],
        # gradient_clip_val=1,
        callbacks=callbacks,
        # accumulate_grad_batches=5,
        precision=16,
        accelerator="gpu", 
        devices=1,
        logger=[wandb_logger],
        fast_dev_run=config["fast_dev_run"]
        )
    tuner = Tuner(trainer)

    if config.get('fine_tune_lr'):
        # Run learning rate finder
        lr_finder = tuner.lr_find(model, dm)
        fig = lr_finder.plot(suggest=True)
        fig.savefig('data/lr_finder.png')
        print("Created `data/lr_finder.png` and stopping.")
        exit()

    trainer.fit(model,
                dm,
                ckpt_path=config.get('resume_trainig_ckpt'))

def parse_args():
    parser = ArgumentParser()

    # Option 1:
    parser.add_argument("--hyperparameter_tune", action='store_true',
                        help="If invoked, all other cli options don't matter.")

    # Option 2:
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--hidden_layer_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    
    parser.add_argument("--ckpt", type=str, help='Load weights for model from ckpt file')
    parser.add_argument("--resume_training_ckpt", type=str, help='Restores full training from ckpt file, not just loading weights')
    
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--deactivate_bert_learning", action='store_true')    
    parser.add_argument("--fast_dev_run", action='store_true')
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--fine_tune_lr", action='store_true', 
                        help="Creates data/lr_finder.png and stops afterwards in order to inspect learning_rates")
    
    return parser.parse_args()

if __name__ == "__main__":
    print("Training module started...")
    args = parse_args()
    args_dict = vars(args)
    
    if args.hyperparameter_tune:
        import yaml
        print("Starting hyperparameter search.")
        sweep_config = yaml.safe_load(open("src/sweep_config.yaml"))
        sweep_id = wandb.sweep(sweep=sweep_config)
        wandb.agent(sweep_id=sweep_id, function=train_func, count=4)
    else:
        print("Start normal training")
        print(f"{args_dict=}")
        train_func(config=args_dict)

