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

from src.config import RegressorConfig, ClassificationConfig
from src.config import config as DATA_CONFIG
from src.model.data_loading import CustomDataModule
from src.model.neural_network import get_model

warnings.filterwarnings("ignore", category=UserWarning)

torch.manual_seed(2024)  

pt_version = torch.__version__
print(f"[INFO] Current PyTorch version: {pt_version} (should be 2.x+)")
print(f'{torch.cuda.is_available()=}')
torch.set_float32_matmul_precision('high')

WANDB_PROJECT = 'news_trading'

def train_func(config: dict = None):
    tensor_logger = TensorBoardLogger("tb_logs", 
                                      name="my_model")
    loggers = [tensor_logger]
    if not config.get('lr_finder'):
        run = wandb.init(save_code=True)
        if config is None: 
            config = wandb.config
        else: 
            wandb.config.update(config)
            
        print(f"{run.settings.mode=}")
        model_config = RegressorConfig if config['task'] == "regression" else ClassificationConfig
        print(model_config)
        print(config)
        
        wandb_logger = WandbLogger(log_model=False, 
                                group='group',
                                offline=True)
        loggers.append(wandb_logger)
    
    model_args = dict((k, config[k]) for k in ('deactivate_bert_learning', 
                                               'learning_rate', 
                                               'dropout_rate', 
                                               'hidden_layer_size',
                                               'n_warm_up_epochs'))
        
    dm = CustomDataModule(news_data_path=DATA_CONFIG.data.learning_dataset, 
                          input_ids_path=model_config.input_ids, 
                          masks_path=model_config.masks, 
                          batch_size=config["batch_size"],
                          target_col_name=model_config.target_col_name)
    
    model: pl.LightningModule = get_model(config.get('ckpt'), model_args, dm, model_config=model_config)

    callbacks = [
        LearningRateMonitor(logging_interval='step',
                            log_momentum=True),
        # StochasticWeightAveraging(swa_lrs=1e-2),
        ]
    
    if not config.get('lr_finder'):
        print(f"ModelCheckpoint path at: data/ckpts/{run.id}")
        callbacks.append(ModelCheckpoint(
                            dirpath=f"data/ckpts/{run.id}",
                            monitor="val/loss",
                            mode="min",
                            save_top_k=1,
                            save_last=False)
                         )

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        # gradient_clip_val=0.1,
        callbacks=callbacks,
        #accumulate_grad_batches=1,
        # precision=16,
        val_check_interval=0.33,
        accelerator="gpu" if not (config["fast_dev_run"] or config["cpu"]) else "cpu", 
        devices=1,
        logger=loggers,
        fast_dev_run=config["fast_dev_run"],
        overfit_batches=config['overfit_batches'] if config['overfit_batches'] else 0.0,
        )
    
    #wandb_logger.watch(model)

    tuner = Tuner(trainer)  

    if config.get('lr_finder'):
        # Run learning rate finder
        lr_finder = tuner.lr_find(model, dm)
        fig = lr_finder.plot(suggest=True)
        fig.savefig('data/lr_finder.png')
        print("Created `data/lr_finder.png` and stopping.")
        exit()

    # Performs one evaluation epoch before starting to train
    # trainer.validate(model, 
    #                  dm,
    #                  ckpt_path=config.get('resume_training_ckpt'))
    trainer.fit(model,
                dm,
                ckpt_path=config.get('resume_training_ckpt'))

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--hidden_layer_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--deactivate_bert_learning", action='store_true')    
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--ckpt", type=str, help='Load weights for model from ckpt file')
    parser.add_argument("--n_warm_up_epochs", type=int, default=1)
    parser.add_argument("--overfit_batches", type=float, default=None,
                        help="pct of samples to fit on. Used for debugging. No validation")
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--task", type=str, help='Either classification or regression')

    # Rare/Optional
    parser.add_argument("--fast_dev_run", action='store_true')
    parser.add_argument("--lr_finder", action='store_true', 
                        help="Creates data/lr_finder.png and stops afterwards in order to inspect learning_rates")
    parser.add_argument("--resume_training_ckpt", type=str, help='Restores full training from ckpt file, not just loading weights')
 
    return parser.parse_args()


if __name__ == "__main__":
    print("Training module started...")
    args = parse_args()
    args_dict = vars(args)
    
    print("Start normal training")
    train_func(config=args_dict)

