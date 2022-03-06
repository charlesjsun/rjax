from typing import Any, Callable, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from rjax.agents.model import Model
from rjax.agents.optimizer import Optimizer
from rjax.common import (PMAP_AXIS_NAME, DeviceList, InfoDict, PRNGKey,
                         UpdateFunction)
from rjax.datasets.dataloader import RandomBatchDataloader


def _shard_batch(x: jnp.ndarray, devices: DeviceList) -> jnp.ndarray:
    device_count = len(devices)
    x = x.reshape((device_count, x.shape[0] // device_count) + x.shape[1:])
    return jax.device_put_sharded(list(x), devices)


class Learner:
    """ Base learner class that takes care of multi-devices training """

    def __init__(self, devices: DeviceList, dataloader: RandomBatchDataloader,
                 train_state_cls: type, replicated: Dict[str, jnp.ndarray],
                 rngs: Dict[str, PRNGKey], update_fn: UpdateFunction,
                 static_argnums: Sequence[int]) -> None:
        """ train_state_cls: the class to make the trainstate from
            replicated: the train_state entries that are not rngs and thus should be replicated.
            rngs: the train_state entries that are rngs and should be split and sharded.
        """
        self.devices = devices
        self.device_count = len(devices)
        self.dataloader = dataloader

        if self.device_count > 1:
            replicated = jax.tree_map(
                lambda x: jax.device_put_replicated(x, devices), replicated)
            rngs = jax.tree_map(
                lambda rng: jax.device_put_sharded(
                    list(jax.random.split(rng, len(devices))), devices), rngs)
        else:
            replicated = {
                k: v.replace(pmapped=False) if isinstance(v, Optimizer) else v
                for k, v in replicated.items()
            }
            replicated = jax.tree_map(lambda x: jax.device_put(x, devices[0]),
                                      replicated)
            rngs = jax.tree_map(lambda x: jax.device_put(x, devices[0]), rngs)
        self.train_state = train_state_cls(**replicated, **rngs)

        if self.device_count > 1:
            self.update_fn = jax.pmap(
                update_fn,
                axis_name=PMAP_AXIS_NAME,
                static_broadcasted_argnums=static_argnums,
            )
        else:
            self.update_fn = jax.jit(update_fn, static_argnums=static_argnums)

    def get_actor(self) -> Model:
        """ Subclasses should always implement this if they want to get an eval actor.
            Simply return the actor model and get_eval_actor() will return an un-sharded version.
        """
        raise NotImplementedError

    def get_eval_actor(self):
        """ Get actor to use for evaluation. Meaning it needs to not be sharded. """
        actor = self.get_actor()
        if self.device_count > 1:
            actor = jax.tree_map(lambda x: x[0], actor)
        return actor

    def sample_batch(self):
        """ Sample a batch and divide it across devices """
        batch = self.dataloader.sample()
        if self.device_count > 1:
            batch = jax.tree_map(lambda x: _shard_batch(x, self.devices),
                                 batch)
        else:
            batch = jax.tree_map(lambda x: jax.device_put(x, self.devices[0]),
                                 batch)
        return batch

    def __call__(self, step: int) -> InfoDict:
        raise NotImplementedError

    def save(self, log_dir: str):
        raise NotImplementedError

    def load(self, log_dir: str):
        raise NotImplementedError
