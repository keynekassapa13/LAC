import random
import os
import numpy as np
import torch

from loguru import logger
from easydict import EasyDict as edict

from dataset import *
from model import *
from utils import parser, util

def main(cfg):
    logger.info(f"Start {cfg.name}...")
    logger.info(f"Config: {cfg.cfg_path}")

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    logger.info(f"Device: {device}")
    cfg.device = device

    logger.info(f"Loading dataset {cfg.data_loader.type} from {cfg.data_loader.data_dir}")
    train_dataset, train_loader = construct_train_loader(cfg)
    if cfg.eval:
        _, train_eval_loader = construct_eval_loader(cfg, pkl_name="train.pkl")
        _, val_eval_loader = construct_eval_loader(cfg, pkl_name="val.pkl")

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Training batch size: {cfg.data_loader.batch_size}")

    if cfg.eval:
        train(cfg, train_loader, train_eval_loader, val_eval_loader)
    else:
        train(cfg, train_loader)


if __name__ == "__main__":
    args = parser.generate_args()
    cfg = parser.load_config(args)

    if ("working_dir" in cfg.trainer) and (cfg.trainer.working_dir != ""):
        cfg.trainer.save_dir = os.path.join(cfg.trainer.working_dir, "models")
        cfg.trainer.log_dir = os.path.join(cfg.trainer.working_dir, "logs")
        cfg.trainer.loguru_dir = os.path.join(cfg.trainer.working_dir, "loguru")

    util.ensure_dir(cfg.trainer.save_dir)
    util.ensure_dir(cfg.trainer.log_dir)
    util.ensure_dir(cfg.trainer.loguru_dir)

    # Add loguru logger
    logger.add(f"{cfg.trainer.loguru_dir}/{cfg.type}-log_trainer.log")

    SEED = 1
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    main(cfg)
