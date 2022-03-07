import contextlib
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import jax
import numpy as np
import tqdm

from rjax.common import DeviceList, InfoDict, PRNGKey
from rjax.experiment.config import BaseConfig
from rjax.experiment.loggers import TensorboardLogger
from rjax.experiment.utils import (create_log_dir_path, get_git_rev,
                                   save_git_diff)

_CallbackType = Callable[[int], Optional[InfoDict]]


@dataclass
class _CheckpointCallback:
    obj: Any
    log_dir: str

    def __call__(self, step: int):
        self.obj.save(self.log_dir, step)


@dataclass
class _CallbackState:
    callback: _CallbackType
    freq: int
    log_freq: int
    log_step: int


class Experiment(contextlib.ExitStack):
    """ A context manager for an experiment """

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self.logging = config.logging

        # a list of context managers
        self.contexts: List[contextlib.AbstractContextManager] = []

        # set random seeds for global seed objects
        np.random.seed(config.seed)
        self.rng = jax.random.PRNGKey(config.seed)

        # setup console logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        formatter = logging.Formatter(
            "[%(asctime)s - %(levelname)s]: %(message)s")
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.INFO)

        # setup various logging
        if self.logging:
            self.log_dir = create_log_dir_path(config)
            os.makedirs(self.log_dir, exist_ok=True)

            # setup logging console to file
            file_handler = logging.FileHandler(
                os.path.join(self.log_dir, "console.log"))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            logging.info(f"Logging to {self.log_dir}")
            logging.info(f'Experiment Config: {config}')

            # Save Config
            with open(os.path.join(self.log_dir, "config.json"), "w") as f:
                config.dump_json(f, indent=4)

            # Save git diff
            if config.git:
                try:
                    commit = get_git_rev().replace(" ", "-")
                    save_git_diff(
                        os.path.join(self.log_dir, f"diff-{commit}.txt"))
                except Exception:
                    logging.warning("Failed to save git diff")

            # setup tensorboard
            if config.tb:
                self.tb_logger = TensorboardLogger(self.log_dir)
                self.register_context(self.tb_logger)

        # checkpointing
        self.checkpoint_freq = self.config.checkpoint_freq
        self.checkpointing = self.logging and self.checkpoint_freq > 0

        # a list of callback states
        self.callback_states: List[_CallbackState] = []

        # setup experiment
        self.max_steps = self.config.max_steps
        self.step = 0

        # setup devices
        all_devices = jax.local_devices()
        self._devices = []
        if len(config.devices) == 0:
            self._devices = all_devices
        else:
            for i in config.devices:
                assert 0 <= i < len(all_devices), (
                    f"there are only {len(all_devices)} available devices")
                self._devices.append(all_devices[i])
        assert len(self._devices) > 0, "there apparently are no devices"

        logging.info(
            f"Running on {len(self._devices)} devices: {self._devices}")

    @property
    def devices(self) -> DeviceList:
        return self._devices

    def split_rng(self) -> PRNGKey:
        key, self.rng = jax.random.split(self.rng)
        return key

    def log(self, infos: Dict[str, Union[np.ndarray, float]]):
        """ Log to all available loggers.
            if infos have a "step" field, use it, otherwise use current step.
        """
        step = infos.get("step", self.step)
        if self.config.tb:
            self.tb_logger.log(infos, step)

    def register_callback(self,
                          callback: _CallbackType,
                          freq: int = 1,
                          log_freq: int = 1):
        """ Registers a callback at a certain frequencey.
            A callback is a callable that takes in the current timestep and optionally returns
            infos to log.
            Optionally can provide a frequency at which to run it at, default 1 (every step).
            Optionally a logging frequency with respect to freq, default 1.
            Ex. freq=2 and log_freq=3 means every third result will be logged (log at every 6 exp steps).
        """
        assert freq > 0 and log_freq > 0
        assert callback is not None
        state = _CallbackState(
            callback=callback,
            freq=freq,
            log_freq=log_freq,
            log_step=1,
        )
        self.callback_states.append(state)

    def clear_callbacks(self):
        """ Remove all step callbacks. Useful for multi-stage experiments that requires multiple
            start() runs with different functions.
        """
        self.callback_states = []

    def start_loop(self, steps: Optional[int] = None):
        """ Run the experiment with the current callbacks for steps amount of steps.
            If steps is None, use config.max_steps.
        """
        if steps is None:
            steps = self.max_steps
        self.step = 0
        for i in tqdm.tqdm(range(1, steps + 1),
                           smoothing=0.1,
                           disable=not self.config.tqdm):
            self.trigger_callbacks()
            self.step = i

    def trigger_callbacks(self):
        """ Triggers all scheduled callbacks. """
        for state in self.callback_states:
            if (self.step + 1) % state.freq == 0:
                infos = state.callback(self.step)
                if (self.step + 1) % state.log_freq == 0:
                    if infos is not None:
                        self.log(infos)
                    state.log_step += 1

    def register_checkpoint(self, obj: Any):
        """ Register an object that has a save method to checkpoint. """
        assert hasattr(obj, "save"), (
            "checkpoint object must have a save(log_dir: str, step: int) method"
        )
        if self.checkpointing:
            self.register_callback(_CheckpointCallback(obj, self.log_dir),
                                   self.checkpoint_freq)

    def register_context(self, cm: contextlib.AbstractContextManager):
        """ Registers a context manager to be entered when this enters a with clause """
        self.contexts.append(cm)

    def clear_contexts(self):
        """ Clear all registered context managers. """
        self.contexts.clear()

    def __enter__(self):
        """ Intentionally left blank. Always call enter_registered_context or enter_context
            INSIDE the with clause. This is so that exceptions during enter_context will be caught by
            __exit__() from inside the with clause.
        """
        return super().__enter__()

    def enter_registered_contexts(self):
        """ Call to enter all registered cm after this has already entered a with clause """
        for cm in self.contexts:
            super().enter_context(cm)

    def enter_context(self, cm: contextlib.AbstractContextManager):
        """ Call to register and enter cm after this has already entered a with clause """
        self.contexts.append(cm)
        super().enter_context(cm)

    def __exit__(self, *args, **kwargs):
        """ Exits all registered context managers """
        super().__exit__(*args, **kwargs)
