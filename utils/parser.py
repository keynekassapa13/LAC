import argparse
import json
from easydict import EasyDict as edict

def generate_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)', required=True)
    return parser.parse_args()

def load_config(args):
    with open(args.config, 'r') as f:
        cfg = edict(json.load(f))
    cfg.cfg_path = args.config
    return cfg