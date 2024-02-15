import torch
import torch.nn as nn
from transformers import BertTokenizerFast
from src.config import config, MODEL_CONFIG
import lightning as pl
from lightning.pytorch import loggers as pl_loggers

from src.model.neural_network import BERTClassifier, initialize_final_layer_bias_with_class_weights
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import StochasticWeightAveraging
from src.model.data_loading import CustomDataModule
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

pt_version = torch.__version__
print(f"[INFO] Current PyTorch version: {pt_version} (should be 2.x+)")
print(f'{torch.cuda.is_available()=}')


# Settings
tokenizer = BertTokenizerFast.from_pretrained(MODEL_CONFIG.tokenizer)
learning_rate = 1e-3 # 5e-5 (slow) for bert, 0.3 (fast) for new feed forward
deactivate_bert_learning = True
ckpt = None
# ckpt = "C:/Users/Adria/Documents/Github Projects/trading_bot/lightning_logs/version_19/checkpoints/epoch=9-step=346.ckpt"


dm = CustomDataModule(news_data_path=config.data.learning_dataset, 
                      input_ids_path=config.data.benzinga.input_ids, 
                      masks_path=config.data.benzinga.masks, 
                      batch_size=512, # Batch size is configured automatically later on
                      target_col_name=MODEL_CONFIG.target_col_name)

if __name__ == "__main__":

    dm.setup("fit")

    class_distribution = dm.get_class_distribution()
    class_weights = 1 / class_distribution.values
    print(dm.get_class_distribution())

    if ckpt:
        model: BERTClassifier = BERTClassifier.load_from_checkpoint(ckpt, deactivate_bert_learning=False)
    else:
        model: BERTClassifier = BERTClassifier(bert_model_name=MODEL_CONFIG.pretrained_network,
                                        num_classes=3,
                                        deactivate_bert_learning=deactivate_bert_learning,
                                        learning_rate=learning_rate,
                                        class_weights=class_weights)
        initialize_final_layer_bias_with_class_weights(model, class_distribution)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="")

    trainer = pl.Trainer(num_sanity_val_steps=2,
                        max_epochs=10,
                        gradient_clip_val=1,
                        callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
                        accumulate_grad_batches=10,
                        precision=16,
                        accelerator="gpu", 
                        devices=1,
                        logger=tb_logger)
    tuner = Tuner(trainer)

    # Auto-scale batch size by growing it exponentially (default)
    tuner.scale_batch_size(model, 
                        mode="power", 
                        datamodule=dm)

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


