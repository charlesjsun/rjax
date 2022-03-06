import os
from typing import Dict, Union

import numpy as np
from tensorboardX import SummaryWriter

from rjax.common import InfoDict


class TensorboardLogger:

    def __init__(self, log_dir: str):
        self.log_dir = log_dir

    def __enter__(self):
        self.summary_writer = SummaryWriter(os.path.join(self.log_dir, 'tb'),
                                            write_to_disk=True).__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        self.summary_writer.__exit__(*args, **kwargs)

    def log(self, infos: InfoDict, step: int):
        for k, v in infos.items():
            if isinstance(v, float) or v.ndim == 0:
                self.summary_writer.add_scalar(f'training/{k}', v, step)
            else:
                self.summary_writer.add_histogram(f'training/{k}', v, step)
        self.summary_writer.flush()
