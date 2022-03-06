from dataclasses import dataclass
from typing import List

from simple_parsing import Serializable, list_field


@dataclass
class BaseConfig(Serializable):
    # set False to disable logging to files
    logging: bool = True
    # how to structure the logging dir
    # delimiters can be '/' to make new directory, or '-' to join strings together with '-'
    # combination of base, env, prefix, name, seed, time (date+time), or config (non-default config)
    logdir_structure: str = "base/env/prefix/config"
    base: str = "logs"
    env: str = ""
    prefix: str = ""
    name: str = ""

    # use tqdm to display experiment loop
    tqdm: bool = True
    # use wandb (check wandb_config.py) for required environment variables
    wandb: bool = False
    # log using TensorboardX
    tb: bool = True
    # saves git diff
    git: bool = True
    # whether to checkpoint models or not
    checkpoint: bool = True

    # random seed
    seed: int = 42
    # maximum number of training steps
    max_steps: int = int(1e6)

    # list of indices for devices to use, leave blank to use all devices
    devices: List[int] = list_field()
