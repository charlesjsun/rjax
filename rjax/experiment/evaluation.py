import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Tuple

import gym
import jax
import numpy as np

import rjax.networks.policies as policies
from rjax.agents.learner import Learner
from rjax.agents.model import Model
from rjax.common import PRNGKey


def _sample_action(rng: PRNGKey, actor: Model,
                   observation: np.ndarray) -> Tuple[PRNGKey, np.ndarray]:
    rng, actions = policies.sample_actions(rng,
                                           actor.apply_fn,
                                           actor.params,
                                           observation,
                                           temperature=0.0)
    actions = np.asarray(actions)
    return rng, np.clip(actions, -1.0, 1.0)


def evaluate(rng: PRNGKey, actor: Model, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    infos = {'return': [], 'length': []}

    for _ in range(num_episodes):
        observation, done = env.reset(), False

        while not done:
            rng, action = _sample_action(rng, actor, observation)
            observation, _, done, info = env.step(action)

        for k in infos.keys():
            infos[k].append(info['episode'][k])

    for k, v in infos.items():
        infos[k] = np.mean(v)

    return infos


@dataclass
class EvalConfig:
    workers: int = 0
    episodes: int = 100
    freq: int = 100000


class EvalManager:

    def __init__(self, rng: PRNGKey, learner: Learner, env: gym.Env,
                 config: EvalConfig) -> None:
        self.rng = rng
        self.learner = learner
        self.env = env
        self.config = config

    def __call__(self, step):
        actor = self.learner.get_eval_actor()
        key, self.rng = jax.random.split(self.rng)
        infos = evaluate(key, actor, self.env, self.config.episodes)
        logging.info(f"step: {step}, eval: {infos}")
        return infos
