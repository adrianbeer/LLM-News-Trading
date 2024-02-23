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
from lightning.pytorch.loggers import WandbLogger

from lightning.pytorch.tuner import Tuner
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
import wandb

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.air.integrations.wandb import WandbLoggerCallback

from src.config import MODEL_CONFIG
from src.config import config as DATA_CONFIG
from src.model.neural_network import get_model
from src.model.data_loading import CustomDataModule

warnings.filterwarnings("ignore", category=UserWarning)

pt_version = torch.__version__
print(f"[INFO] Current PyTorch version: {pt_version} (should be 2.x+)")
print(f'{torch.cuda.is_available()=}')
torch.set_float32_matmul_precision('high')

def train_model(config: dict):
    model_args = dict(
        deactivate_bert_learning=config["deactivate_bert_learning"],
        learning_rate=config["learning_rate"],
        dropout_rate=config["dropout_rate"],
        hidden_layer_size=config["hidden_layer_size"],
    )

    dm = CustomDataModule(news_data_path=DATA_CONFIG.data.learning_dataset, 
                          input_ids_path=DATA_CONFIG.data.news.input_ids, 
                          masks_path=DATA_CONFIG.data.news.masks, 
                          batch_size=config["batch_size"], # Batch size is configured automatically later on
                          target_col_name=MODEL_CONFIG.target_col_name)
    
    model: pl.LightningModule = get_model(config["ckpt"], model_args, dm)

    tb_logger = pl_loggers.TensorBoardLogger('tb_logs', 
                                             name="bert_regressor",
                                             flush_secs=600)
    wandb_logger = WandbLogger(log_model="all", 
                               project='news_trading')
    
    callbacks = [
        RayTrainReportCallback(),
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
                        logger=[tb_logger, wandb_logger],
                        fast_dev_run=config["fast_dev_run"],
                        strategy=RayDDPStrategy(),
                        plugins=[RayLightningEnvironment()])
    trainer = prepare_trainer(trainer)
    tuner = Tuner(trainer)

    if config["batch_size"] is None:
        # Auto-scale batch size by growing it exponentially (default)
        tuner.scale_batch_size(model,
                               mode="power",
                               datamodule=dm)
        dm.batch_size = min(dm.batch_size,  512)

    if config["learning_rate"] is None:
        # Run learning rate finder
        lr_finder = tuner.lr_find(model, dm)
        fig = lr_finder.plot(suggest=True)
        fig.savefig('data/lr_finder.png')
        new_lr = lr_finder.suggestion()
        model.hparams.learning_rate = new_lr
        
    if config["stop_after_lr_finder"]:
        print("Created data/lr_finder.png and stopping.")
        exit

    print(f"Start training with {model.hparams.learning_rate=}")
    trainer.fit(model,
                dm,
                ckpt_path=config["ckpt"])


search_space = {
    "batch_size": tune.choice([32, 64, 128, 512]),
    "hidden_layer_size": tune.choice([10, 128, 786]),
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "ckpt": None,
    "fast_dev_run": False,
    "deactivate_bert_learning": True,
    "dropout_rate": tune.choice([0.1]),
    "stop_after_lr_finder": False,
}

# The maximum training epochs
num_epochs = 2

# Number of sampls from parameter space
num_samples = 1


scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

scaling_config = ScalingConfig(
    num_workers=1, 
    use_gpu=True, 
    resources_per_worker={"CPU": 4, "GPU": 1}
)

run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    ),
    callbacks=[WandbLoggerCallback(project="news_trading",
                                   log_config=True,
                                   api_key='601e267b2d7662e90fd91b4ce60196406c6c86bf')]
)

# Define a TorchTrainer without hyper-parameters for Tuner
ray_trainer = TorchTrainer(
    train_model,
    scaling_config=scaling_config,
    run_config=run_config,
)

def tune_model_asha(num_samples=10):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val_loss", 
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()


#TODO remove argsparser and just use config and ray? try out ray... 
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--hyperparameter_tune", action='store_true',
                        help="If invoked, all other cli options don't matter.")

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
    args_dict = vars(args)
    
    if args.hyperparameter_tune:
        results = tune_model_asha(num_samples=num_samples)
        print(results)
    else:
        train_model(config=args_dict)

