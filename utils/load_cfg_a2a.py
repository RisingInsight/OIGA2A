# --*--coding   : utf-8  --*--
# @Time         : 2026/5/6
# @Author       : RisingInsight
# @File         : load_cfg_a2a.py
# @Software     : PyCharm
# @Project      : OIGA2A-new
# @Function     : Config loader for A2A (Any-to-Any) pipeline
import yaml
import argparse
import os


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    for key, value in params.items():
        if isinstance(value, dict):
            params[key] = argparse.Namespace(**value)
    params = argparse.Namespace(**params)
    if hasattr(params, 'training') and hasattr(params.training, 'optimizer'):
        params.training.optimizer['lr'] = float(params.training.optimizer['lr'])
    return params


def load_datasets_config(datasets):
    params = dict()
    for dataset in datasets:
        cfg_path = os.path.join("configs", "datasets", "FLAIR_A2A", f"{dataset}.yml")
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join("configs", "datasets", "FLAIR_A2A_v2", f"{dataset}.yml")
        params[dataset] = load_config(cfg_path)
    return params
