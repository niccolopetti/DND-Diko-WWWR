import argparse
from easydict import EasyDict
from pathlib import Path
import os
import yaml

project_root = Path(__file__).resolve().parents[1]
data_root = '/media/niccolo/DATA/DND-Diko-WWWR/Challenge/DND-Diko-WWWR'  # path to dataset

cfg = EasyDict()

# COMMON CONFIGS
cfg.CROPTYPE = 'images'
cfg.SEED = 42
cfg.NWORK = 4

# TRAIN CONFIGS
cfg.TRAIN = EasyDict()
# TEST CONFIGS
cfg.TEST = EasyDict()

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r') as f:
        yaml_cfg = EasyDict(yaml.safe_load(f))
    for k, v in yaml_cfg.items():
        if k in cfg:
            if isinstance(cfg[k], dict) and isinstance(v, dict):
                cfg[k].update(v)
            else:
                if isinstance(cfg[k], dict) or isinstance(v, dict):
                    raise TypeError(f"Conflict in key {k}: cfg[k] is {type(cfg[k])} while yaml value is {type(v)}.")
                cfg[k] = v
        else:
            cfg[k] = v
    return cfg


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for ...")
    parser.add_argument('--is_test', action='store_true')
    parser.add_argument('--use_cache', dest='USE_CACHE', help='use preprocessed cache for acceleration', action='store_true')
    parser.add_argument('--p_interval', type=int, default=100)
    parser.add_argument('--ngpus', dest='NGPUS', type=int, default=1)
    parser.add_argument('--im_scale', dest='im_scale', type=int, default=896)
    parser.add_argument('--bsz', type=int, default=4)
    parser.add_argument('--acc_bsz', type=int, default=1)
    parser.add_argument('--cfg', type=str, default='./configs/WW2020.yml', required=True)
    parser.add_argument('--exp_name', type=str, default="play")
    parser.add_argument('--restore_from', type=str, default="")
    parser.add_argument('--codalab_pred', type=str, default="test")
    parser.add_argument('--train_set', type=str, default="trainval")
    parser.add_argument('--use_amp', help='Use automatic mixed precision during training.')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    print(args)
