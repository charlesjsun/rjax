import collections
from typing import Callable, Optional, Tuple

import d4rl
import gym
import numpy as np

from rjax.datasets.d4rl.utils import (get_preprocessing_fn,
                                      sequence_dataset_iter)
from rjax.datasets.dataset import Dataset

Batch = collections.namedtuple(
    'Batch',
    [
        'observations',  # [ batch_size x (seq_len + 1) x obs_dim ]
        'actions',  # [ batch_size x seq_len x act_dim ]
        'rewards',  # [ batch_size x seq_len ]
        'terminals',  # [ batch_size x seq_len ]
        'pad_masks',  # [ batch_size x (seq_len + 1) ]
    ])


class D4RLSequenceDataset(Dataset):

    def __init__(
        self,
        env_name: str,
        env: gym.Env,
        seq_len: int = 15,
        front_pad: int = 0,
        back_pad: int = 0,
    ):

        self.env = env
        self.seq_len = seq_len
        self.front_pad = front_pad
        self.back_pad = back_pad

        dataset = self.env.get_dataset()
        dataset = get_preprocessing_fn(env_name)(dataset)

        dataset_iter = sequence_dataset_iter(self.env, dataset)

        self.path_lengths = []
        self.observations_segmented = []
        self.actions_segmented = []
        self.rewards_segmented = []
        self.terminals_segmented = []
        self.pad_masks_segmented = []

        for data in dataset_iter:
            assert data["steps"] == data["rewards"].shape[0]
            assert data["steps"] + 1 == data["observations"].shape[0]
            self.path_lengths.append(data["steps"])
            self.observations_segmented.append(data["observations"].astype(
                np.float32))
            self.actions_segmented.append(data["actions"].astype(np.float32))
            self.rewards_segmented.append(data["rewards"].astype(np.float32))
            self.terminals_segmented.append(data["terminals"].astype(
                np.float32))
            self.pad_masks_segmented.append(
                np.ones(data["observations"].shape[0], dtype=np.float32))

        self.n_trajectories = len(self.path_lengths)

        # padding
        for i in range(self.n_trajectories):
            self.path_lengths[
                i] = self.front_pad + self.path_lengths[i] + self.back_pad
            self.observations_segmented[i] = np.pad(
                self.observations_segmented[i],
                ((self.front_pad, self.back_pad), (0, 0)),
                constant_values=0.0)
            self.actions_segmented[i] = np.pad(
                self.actions_segmented[i],
                ((self.front_pad, self.back_pad), (0, 0)),
                constant_values=0.0)
            self.rewards_segmented[i] = np.pad(
                self.rewards_segmented[i], ((self.front_pad, self.back_pad), ),
                constant_values=0.0)
            self.terminals_segmented[i] = np.pad(
                self.terminals_segmented[i],
                ((self.front_pad, self.back_pad), ),
                constant_values=0.0)
            self.pad_masks_segmented[i] = np.pad(
                self.pad_masks_segmented[i],
                ((self.front_pad, self.back_pad), ),
                constant_values=0.0)

        # generate dataset indices
        indices = []
        for path_ind, length in enumerate(self.path_lengths):
            end = length - self.seq_len + 1
            for i in range(end):
                indices.append((path_ind, i, i + self.seq_len))

        self.indices = np.array(indices)
        self._size = len(self.indices)

    @property
    def size(self) -> int:
        return self._size

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        path_ind, start_ind, end_ind = self.indices[idx]

        # [ (seq_len + 1) x obs_dim ]
        observations = self.observations_segmented[path_ind][
            start_ind:end_ind + 1]
        # [ seq_len x act_dim ]
        actions = self.actions_segmented[path_ind][start_ind:end_ind]
        # [ seq_len ]
        rewards = self.rewards_segmented[path_ind][start_ind:end_ind]
        # [ seq_len ]
        terminals = self.terminals_segmented[path_ind][start_ind:end_ind]
        # [ (seq_len + 1) ]
        pad_masks = self.pad_masks_segmented[path_ind][start_ind:end_ind + 1]

        return observations, actions, rewards, terminals, pad_masks

    def get_random_batch(self, batch_size: int) -> Batch:
        indices = np.random.randint(self.size, size=batch_size)

        observations, actions, rewards, terminals, pad_masks = zip(
            *[self[idx] for idx in indices])

        return Batch(observations=np.stack(observations, axis=0),
                     actions=np.stack(actions, axis=0),
                     rewards=np.stack(rewards, axis=0),
                     terminals=np.stack(terminals, axis=0),
                     pad_masks=np.stack(pad_masks, axis=0))
