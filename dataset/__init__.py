from .pouring import Pouring
from .pennaction import PennAction
from torch.utils.data import DataLoader

def construct_train_loader(cfg, pkl_name="train.pkl"):
    if cfg.data_loader.type.lower() == "pouring":
        dataset = Pouring(cfg, pkl_name=pkl_name)
    elif cfg.data_loader.type.lower() == "pennaction":
        dataset = PennAction(cfg, 
                             pkl_name=pkl_name,
                             action=cfg.data_loader.action)

    loader = DataLoader(
        dataset,
        batch_size=cfg.data_loader.batch_size,
        shuffle=cfg.data_loader.shuffle,
        num_workers=cfg.data_loader.num_workers,
        pin_memory=True,
        drop_last=True)
    return dataset, loader

def construct_eval_loader(cfg, pkl_name="val.pkl"):
    if cfg.data_loader.type.lower() == "pouring":
        dataset = Pouring(cfg, pkl_name=pkl_name, mode="eval")
    elif cfg.data_loader.type.lower() == "pennaction":
        dataset = PennAction(cfg, 
                             pkl_name=pkl_name, 
                             mode="eval", 
                             action=cfg.data_loader.action)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data_loader.num_workers,
        pin_memory=True,
        sampler=None) 
    return dataset, loader