import argparse
import json
from easydict import EasyDict as edict
import os
from loguru import logger

def generate_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)', required=True)
    parser.add_argument('-n', '--name', default=None, type=str, help='name of the experiment')
    return parser.parse_args()

def load_config(args):
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    with open(args.config, 'r') as f:
        cfg = edict(json.load(f))
    cfg.cfg_path = args.config
    if args.name:
        cfg.name = args.name
    return cfg