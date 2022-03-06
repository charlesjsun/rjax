from typing import Tuple

import gym

from rjax.environments.episode_monitor import EpisodeMonitor
from rjax.environments.single_precision import SinglePrecision


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)

    env = EpisodeMonitor(env)
    env = SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env
