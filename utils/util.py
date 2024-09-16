import json
from loguru import logger
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import os

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

# ========================================================================

def load_ckpt(cfg, model, optimizer, base_path="./"):
    logger.info(f"Loding checkpoint from {cfg.trainer.save_dir}")
    path = base_path + cfg.trainer.save_dir
    if os.path.exists(path):
        model_names = [m for m in os.listdir(path) if m.startswith(cfg.type) and m.endswith(".pth")]
        if len(model_names) > 0:
            logger.info(f"Model names: {model_names}")
            model_names.sort()
            def sort_key(filename):
                return int(filename.split("_")[-1].split(".")[0])
            model_names.sort(key=sort_key) 
            logger.info(f"Model names: {model_names}")
            model_name = model_names[-1]
            logger.info(f"Loading model: {model_name}")
            path = os.path.join(path, model_name)
            logger.info(f"Loading model from {path}")
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded checkpoint from {path}")
            return model, optimizer, checkpoint["epoch"]
        else:
            logger.info(f"No model found in {path}")
    else:
        logger.info(f"Path {path} does not exist")
        
    return model, optimizer, 0