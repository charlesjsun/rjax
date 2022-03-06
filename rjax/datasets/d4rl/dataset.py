import collections
from typing import Tuple

import d4rl
import gym
import numpy as np

from rjax.datasets.d4rl.utils import get_preprocessing_fn
from rjax.datasets.dataset import Dataset

Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "terminals", "next_observations"])


class D4RLDataset(Dataset):

    def __init__(self, env_name: str, env: gym.Env):
        dataset = d4rl.qlearning_dataset(env)
        dataset = get_preprocessing_fn(env_name)(dataset)

        dones_float = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset["observations"][i + 1] -
                              dataset["next_observations"][i]
                              ) > 1e-6 or dataset["terminals"][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        self.observations = dataset["observations"].astype(np.float32)
        self.actions = dataset["actions"].astype(np.float32)
        self.rewards = dataset["rewards"].astype(np.float32)
        self.terminals = dataset["terminals"].astype(np.float32)
        self.next_observations = dataset["next_observations"].astype(
            np.float32)
        self.dones_float = dones_float.astype(np.float32)
        self._size = len(dataset["observations"])

    @property
    def size(self) -> int:
        return self._size

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (self.observations[idx], self.actions[idx], self.rewards[idx],
                self.terminals[idx], self.next_observations[idx])

    def get_random_batch(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     terminals=self.terminals[indx],
                     next_observations=self.next_observations[indx])
