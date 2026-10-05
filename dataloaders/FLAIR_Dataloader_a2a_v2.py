# --*--coding   : utf-8  --*--
# @File         : FLAIR_Dataloader_a2a_v2.py
# @Function     : A2A V2 数据加载器，支持开放指令文本控制，返回 (source, target, src_id, tgt_id, text_tokens)
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2
import torch
import os
import numpy as np
import webdataset as wds
import glob
import json
from torch.utils.data import IterableDataset
from utils.load_cfg_a2a import load_datasets_config
import io
import tifffile
import random
import clip

from dataloaders.instruction_generator import InstructionGenerator


class GenericDataModule_V2(pl.LightningDataModule):
    def __init__(self,
                 datasets_folder="./datasets_preprocess/",
                 train_batch_size=8,
                 test_batch_size=16,
                 train_image_size=(512, 512),
                 num_workers=4,
                 dataset_names=None,
                 train_cfg_training=None,
                 mixed_precision=False,
                 modality_pairs=None,
                 clip_model_name="ViT-L/14",
                 lora_rank=4,
                 lora_alpha=4,
                 instructions_path=None):
        super().__init__()
        self.datasets_folder = datasets_folder
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_image_size = train_image_size
        self.num_workers = num_workers

        self.train_dataset_names = dataset_names.train_datasets
        self.val_dataset_names = dataset_names.val_datasets
        self.test_dataset_names = dataset_names.test_datasets

        self.train_datasets_cfg = load_datasets_config(self.train_dataset_names)
        self.val_datasets_cfg = load_datasets_config(self.val_dataset_names)
        self.test_datasets_cfg = load_datasets_config(self.test_dataset_names)
        self.train_cfg_training = train_cfg_training
        if not modality_pairs:
            raise ValueError("modality_pairs must be a non-empty list of [src_id, tgt_id] pairs")
        self.modality_pairs = modality_pairs

        self.clip_model_name = clip_model_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.instructions_path = instructions_path

        # Modality ID to name mapping
        self.modality_names = {0: "rgb", 1: "nir", 2: "dem", 3: "spot"}

        self.train_transform = v2.Compose([
            v2.RandomResizedCrop(self.train_image_size,
                                 scale=(0.5, 1.0),
                                 interpolation=v2.InterpolationMode.BILINEAR,
                                 antialias=True),
            v2.ToDtype(torch.float16, scale=True) if mixed_precision else v2.ToDtype(torch.float32, scale=True),
        ])
        self.val_transform = v2.Compose([
            v2.Resize(self.train_image_size, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.ToDtype(torch.float16, scale=True) if mixed_precision else v2.ToDtype(torch.float32, scale=True),
        ])

        self.train_loader_config = {
            'batch_size': self.train_batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': None,
        }
        self.eval_loader_config_generic = {
            'batch_size': self.test_batch_size,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': None,
        }
        self.eval_loader_config_generic_diverse = {
            'batch_size': 1,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': None,
        }
        self.save_hyperparameters()

    def setup(self, stage):
        # Initialize instruction generator (text encoding happens inside model forward)
        self.instruction_generator = InstructionGenerator(instructions_path=self.instructions_path)

        # Modality pairs as list of tuples for random.choice
        self.modality_pairs_list = [(p[0], p[1]) for p in self.modality_pairs]

        if stage == 'fit':
            self.train_datasets = []
            for dataset_name in self.train_dataset_names:
                dataset_configs = self.train_datasets_cfg[dataset_name].train
                dataset = self.get_generic_dataset('train', dataset_configs, self.train_transform,
                                                    dataset_name,
                                                    num_samples_per_epoch=self.train_cfg_training.num_samples_per_epoch)
                self.train_datasets.append(dataset)

        if stage == 'fit' or stage == 'validate':
            self.val_datasets = []
            for dataset_name in self.val_dataset_names:
                dataset_configs = self.val_datasets_cfg[dataset_name].val
                dataset = self.get_generic_dataset('val', dataset_configs, self.val_transform, dataset_name)
                self.val_datasets.append(dataset)

        if stage == 'fit' or stage == 'test':
            self.test_datasets = []
            for dataset_name in self.test_dataset_names:
                dataset_configs = self.test_datasets_cfg[dataset_name].test
                dataset = self.get_generic_dataset('test', dataset_configs, self.val_transform, dataset_name)
                self.test_datasets.append(dataset)
        print({'train_datasets': self.train_datasets_cfg,
               'val_datasets': self.val_datasets_cfg,
               'test_datasets': self.test_datasets_cfg})

    def get_generic_dataset(self,
                            split,
                            dataset_configs,
                            transform,
                            dataset_name,
                            num_samples_per_epoch=None):

        dataset_list = []
        for dataset_config in dataset_configs:
            allshards = glob.glob(os.path.join(self.datasets_folder, dataset_config["datafolder_name"], "dataset-*.tar"))
            dataset = wds.WebDataset(allshards,
                                     nodesplitter=wds.split_by_node,
                                     resampled=True if split == 'train' else False,
                                     shardshuffle=True if split == 'train' else False)
            dataset = dataset.rename(
                rgb='rgb.tif',
                nir='nir.tif',
                dem='dem.tif',
                spot='spot.tif'
            )
            dataset = dataset.to_tuple('rgb', 'nir', 'dem', 'spot', '__key__')
            if split == 'train':
                dataset = (dataset
                           .shuffle(1000)
                           .map(lambda sample: self.a2a_preprocess(sample, transform))
                           .with_epoch(num_samples_per_epoch)
                           .with_length(num_samples_per_epoch))
            else:
                metadata_path = os.path.join(self.datasets_folder, dataset_config["datafolder_name"], "metadata.json")
                with open(metadata_path) as f:
                    metadata = json.load(f)
                dataset = (dataset
                           .map(lambda sample: self.a2a_preprocess(sample, transform))
                           .with_length(metadata["total_samples"]))
                dataset.total_len = metadata["total_samples"]

            dataset.split = split
            dataset.dataset_name = dataset_name
            dataset.my_shard_num = len(allshards)
            dataset_list.append(dataset)
            try:
                print(f"< {dataset.__class__.__name__}, {dataset.dataset_name} - #database: {len(dataset)}; #queries: {len(dataset)} >")
            except TypeError as e:
                print(f"< {dataset.__class__.__name__}, {dataset.dataset_name} - #database: {dataset.total_len}; #queries: {dataset.total_len} >")

        if len(dataset_list) > 1:
            if split == 'train':
                dataset = RandomChainDataset(dataset_list, num_samples_per_epoch=num_samples_per_epoch)
                dataset.dataset_name = dataset_name
                dataset.split = dataset_list[-1].split
            else:
                dataset = torch.utils.data.ChainDataset(dataset_list)
                dataset.dataset_name = dataset_name
                dataset.split = dataset_list[-1].split
        else:
            dataset = dataset_list[0]
        if "diverse_size" in dataset_config and dataset_config["diverse_size"]:
            dataset.diverse_size = True
        return dataset

    def a2a_preprocess(self, sample, transform):
        """A2A V2 预处理：随机选择 src/tgt 对，生成指令并 tokenize"""
        rgb_bytes, nir_bytes, dem_bytes, spot_bytes, key = sample
        modality_bytes = {0: rgb_bytes, 1: nir_bytes, 2: dem_bytes, 3: spot_bytes}
        src_id, tgt_id = random.choice(self.modality_pairs_list)
        source = self._decode_and_transform(modality_bytes[src_id], transform)
        target = self._decode_and_transform(modality_bytes[tgt_id], transform)

        # Generate instruction and tokenize (encoding happens inside model forward)
        src_name = self.modality_names[src_id]
        tgt_name = self.modality_names[tgt_id]
        instruction = self.instruction_generator.generate(src_name, tgt_name)
        tokens = clip.tokenize(instruction, truncate=True).squeeze(0)  # (77,)

        return source, target, src_id, tgt_id, tokens

    def _decode_and_transform(self, raw_bytes, transform):
        """解码 TIFF 字节并应用变换"""
        img_np = tifffile.imread(io.BytesIO(raw_bytes))
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        return transform(img_tensor)

    def train_dataloader(self):
        mixed_dataset = RandomChainDataset(self.train_datasets,
                                           num_samples_per_epoch=self.train_cfg_training.num_samples_per_epoch)
        train_dataloaders = DataLoader(dataset=mixed_dataset, **self.train_loader_config)
        return train_dataloaders

    def val_dataloader(self):
        val_dataloaders = []
        for index, val_dataset_name in enumerate(self.val_dataset_names):
            val_dataset = self.val_datasets[index]
            val_dataloaders.append(DataLoader(dataset=val_dataset, num_workers=max(min(val_dataset.my_shard_num,
                                              self.num_workers)//len(self.trainer.device_ids), 1),
                                              **self.eval_loader_config_generic))
        return val_dataloaders

    def test_dataloader(self):
        test_dataloaders = []
        for index, test_dataset_name in enumerate(self.test_dataset_names):
            test_dataset = self.test_datasets[index]
            if hasattr(test_dataset, "diverse_size") and test_dataset.diverse_size:
                test_dataloaders.append(DataLoader(
                        dataset=test_dataset, num_workers=max(min(test_dataset.my_shard_num, self.num_workers)//len(self.trainer.device_ids), 1), **self.eval_loader_config_generic_diverse))
            else:
                test_dataloaders.append(DataLoader(
                    dataset=test_dataset, num_workers=max(min(test_dataset.my_shard_num, self.num_workers)//len(self.trainer.device_ids), 1), **self.eval_loader_config_generic))
        return test_dataloaders

    def generate_dataloader(self, dataset_names):
        """为推理/生成任务返回 DataLoader 列表"""
        dataloaders = []
        for ds_name in dataset_names:
            try:
                idx = self.test_dataset_names.index(ds_name)
            except ValueError:
                raise ValueError(
                    f"Dataset '{ds_name}' not found in test_datasets! "
                    f"Available: {self.test_dataset_names}"
                )
            test_dataset = self.test_datasets[idx]

            if hasattr(test_dataset, "diverse_size") and test_dataset.diverse_size:
                loader_cfg = self.eval_loader_config_generic_diverse
            else:
                loader_cfg = self.eval_loader_config_generic

            shard_num = getattr(test_dataset, 'my_shard_num', 1)
            if hasattr(self, 'trainer') and self.trainer is not None:
                num_workers = max(
                    min(shard_num, self.num_workers) // len(self.trainer.device_ids), 1
                )
            else:
                num_workers = min(shard_num, self.num_workers)

            print(f"Dataset {ds_name}: shard_num={shard_num}, using num_workers={num_workers}")

            dl = DataLoader(
                dataset=test_dataset,
                num_workers=num_workers,
                **loader_cfg
            )
            dataloaders.append(dl)
        return dataloaders


class RandomChainDataset(torch.utils.data.ChainDataset):
    """自定义随机链式数据集，从多个数据源中随机采样"""
    def __init__(self, datasets, num_samples_per_epoch=None, probs=None, longest=True):
        super().__init__(datasets)
        self.probs = probs
        self.longest = longest
        self.num_samples_per_epoch = num_samples_per_epoch

    def __iter__(self):
        sources = [iter(d) for d in self.datasets]
        return self.random_samples(sources, self.probs, longest=self.longest)

    def random_samples(self, sources, probs=None, longest=False):
        if probs is None:
            probs = [1] * len(sources)
        else:
            probs = list(probs)

        while len(sources) > 0:
            cum = (np.array(probs) / np.sum(probs)).cumsum()
            r = torch.rand(1).item()
            i = np.searchsorted(cum, r)
            try:
                yield next(sources[i])
            except StopIteration:
                if longest:
                    del sources[i]
                    del probs[i]
                else:
                    break

    def __len__(self):
        if self.num_samples_per_epoch is None:
            total = 0
            for d in self.datasets:
                assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
                total += len(d)
            return total
        else:
            return self.num_samples_per_epoch