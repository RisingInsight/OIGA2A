# --*--coding   : utf-8  --*--
# @File         : main_a2a_v2.py
# @Function     : A2A V2 模型训练 - 支持开放指令文本控制 (CLIP LoRA + cross-attention)

import pytorch_lightning as pl

from swanlab.integration.pytorch_lightning import SwanLabLogger
from pytorch_lightning.strategies import DDPStrategy
import argparse
from oiga2a_model import OIGA2A
from utils.load_cfg_a2a import load_config
from dataloaders.FLAIR_Dataloader_a2a_v2 import GenericDataModule_V2
import torch
torch.serialization.add_safe_globals([argparse.Namespace])

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':
    pl.seed_everything(3407, workers=True)
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str)
    args = args.parse_args()

    train_cfg = load_config(args.config)
    swanlab_logger = SwanLabLogger(name=args.config.split('/')[-1].split('.')[0],
                                    entity="unistgl", project="OIGA2A_A2A_V2")
    modality_pairs = getattr(train_cfg.datasets, 'modality_pairs',
                             [[0, 1], [0, 2], [0, 3], [1, 0], [2, 0], [3, 0]])

    datamodule = GenericDataModule_V2(
        datasets_folder=train_cfg.datasets.datasets_folder,
        train_batch_size=train_cfg.training.train_batch_size,
        test_batch_size=train_cfg.training.test_batch_size,
        train_image_size=train_cfg.training.train_image_size,
        num_workers=train_cfg.training.num_workers,
        dataset_names=train_cfg.datasets,
        train_cfg_training=train_cfg.training,
        mixed_precision=True if train_cfg.training.mixed_precision else False,
        modality_pairs=modality_pairs,
        instructions_path=getattr(train_cfg.datasets, 'instructions_path', None),
    )

    training_stage = train_cfg.training.stage if hasattr(train_cfg.training, 'stage') else 1

    if hasattr(train_cfg.training, "load") and train_cfg.training.load_type == "finetune":
        model = OIGA2A.load_from_checkpoint(train_cfg.training.load,
                                                    model_arch=train_cfg.model.model_arch,
                                                    model_config=train_cfg.model.model_config,
                                                    lr=train_cfg.training.optimizer["lr"],
                                                    optimizer=train_cfg.training.optimizer["name"],
                                                    weight_decay=train_cfg.training.optimizer["weight_decay"],
                                                    momentum=train_cfg.training.optimizer["momentum"],
                                                    lr_sched=train_cfg.training.scheduler["name"],
                                                    lr_sched_args=train_cfg.training.scheduler["args"],
                                                    loss_name=train_cfg.training.loss["name"],
                                                    loss_config=train_cfg.training.loss["config"],
                                                    training_stage=training_stage,
                                                    gradient_accumulation=train_cfg.training.gradient_accumulation if hasattr(train_cfg.training, "gradient_accumulation") else 1,
                                                    gradient_clip_val=train_cfg.training.gradient_clip_val if hasattr(train_cfg.training, "gradient_clip_val") else 0.0,
                                                    gradient_clip_algorithm=train_cfg.training.gradient_clip_algorithm if hasattr(train_cfg.training, "gradient_clip_algorithm") else 'norm',
                                                    strict=False)
    else:
        model = OIGA2A(
            model_arch=train_cfg.model.model_arch,
            model_config=train_cfg.model.model_config,
            lr=train_cfg.training.optimizer["lr"],
            optimizer=train_cfg.training.optimizer["name"],
            weight_decay=train_cfg.training.optimizer["weight_decay"],
            momentum=train_cfg.training.optimizer["momentum"],
            lr_sched=train_cfg.training.scheduler["name"],
            lr_sched_args=train_cfg.training.scheduler["args"],
            loss_name=train_cfg.training.loss["name"],
            loss_config=train_cfg.training.loss["config"],
            training_stage=training_stage,
            gradient_accumulation=train_cfg.training.gradient_accumulation if hasattr(train_cfg.training,
                                                                                      "gradient_accumulation") else 1,
            calculate_stats=train_cfg.training.calculate_stats if hasattr(train_cfg.training,
                                                                          "calculate_stats") else False,
            gradient_clip_val=train_cfg.training.gradient_clip_val if hasattr(train_cfg.training, "gradient_clip_val") else 0.0,
            gradient_clip_algorithm=train_cfg.training.gradient_clip_algorithm if hasattr(train_cfg.training, "gradient_clip_algorithm") else 'norm',
        )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        filename=f'{model.model_arch}' + '_{epoch:02d}',
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_on_train_epoch_end=True,
        save_last=True,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        accelerator='gpu',
        devices="auto",
        default_root_dir=f'./logs/',
        num_nodes=1,
        num_sanity_val_steps=0,
        precision="16-mixed" if train_cfg.training.mixed_precision else "32-true",
        max_epochs=train_cfg.training.num_epochs,
        limit_val_batches=0,
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=20,
        strategy=DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else "auto",
        logger=swanlab_logger,
    )

    if hasattr(train_cfg.training, "load"):
        print(f"Loading Model: {train_cfg.training.load}")
        if train_cfg.training.load_type == "resume":
            print("RESUME FROM CKPT")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=train_cfg.training.load)
        elif train_cfg.training.load_type == "finetune":
            print("FINETUNE FROM CKPT")
            trainer.fit(model=model, datamodule=datamodule)
    else:
        print("Training from scratch")
        trainer.fit(model=model, datamodule=datamodule)