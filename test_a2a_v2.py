# --*--coding   : utf-8  --*--
# @File         : test_a2a_v2.py
# @Function     : A2A V2 模型测试 - 支持开放指令文本控制
import pytorch_lightning as pl
import argparse
import torch
torch.serialization.add_safe_globals([argparse.Namespace])
from swanlab.integration.pytorch_lightning import SwanLabLogger
from oiga2a_model import OIGA2A
from utils.load_cfg_a2a import load_config
from dataloaders.FLAIR_Dataloader_a2a_v2 import GenericDataModule_V2

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':
    pl.seed_everything(3407, workers=True)
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str)
    args = args.parse_args()
    test_cfg = load_config(args.config)
    config_name = args.config.split('/')[-1].split('.')[0]
    swanlab_logger = SwanLabLogger(
        name=f"test_{config_name}",
        entity="unistgl",
        project="OIGA2A_A2A_V2",
    )
    modality_pairs = getattr(test_cfg.datasets, 'modality_pairs', [[0, 1], [0, 2], [0, 3], [1, 0], [2, 0], [3, 0]])
    datamodule = GenericDataModule_V2(
        datasets_folder=test_cfg.datasets.datasets_folder,
        train_batch_size=test_cfg.training.train_batch_size,
        test_batch_size=test_cfg.training.test_batch_size,
        train_image_size=test_cfg.training.train_image_size,
        num_workers=test_cfg.training.num_workers,
        dataset_names=test_cfg.datasets,
        train_cfg_training=test_cfg.training,
        mixed_precision=False,
        modality_pairs=modality_pairs,
        instructions_path=getattr(test_cfg.datasets, 'instructions_path', None),
    )
    model = OIGA2A.load_from_checkpoint(
        test_cfg.training.load,
        model_arch=test_cfg.model.model_arch,
        model_config=test_cfg.model.model_config,
    )
    trainer = pl.Trainer(
        devices="auto",
        strategy="auto",
        logger=swanlab_logger,
    )
    trainer.test(model, datamodule=datamodule)