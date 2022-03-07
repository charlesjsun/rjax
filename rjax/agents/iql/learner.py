"""Implementations of algorithms for continuous control."""

import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Sequence, Tuple

import flax
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training import checkpoints
from simple_parsing import list_field

import rjax.networks.policies as policies
from rjax.agents.iql.actor import update as awr_update_actor
from rjax.agents.iql.critic import update_q, update_target, update_v
from rjax.agents.learner import PMAP_AXIS_NAME, Learner
from rjax.agents.model import Model
from rjax.agents.optimizer import Optimizer
from rjax.common import DeviceList, InfoDict, PRNGKey
from rjax.datasets.d4rl.dataset import Batch
from rjax.datasets.dataloader import RandomBatchDataloader
from rjax.networks.value_net import DoubleQNet, ValueNet


@dataclass
class IQLConfig:
    actor_lr: float = 3e-4
    v_lr: float = 3e-4
    q_lr: float = 3e-4
    hidden_dims: List[int] = list_field(256, 256)
    discount: float = 0.99
    tau: float = 0.005
    expectile: float = 0.9
    temperature: float = 10.0
    dropout_rate: Optional[float] = None
    opt_decay_schedule: str = "cosine"
    log_freq: int = 1000


@struct.dataclass
class TrainState:
    actor: Model
    q_net: Model
    v_net: Model
    target_q_net: Model

    actor_opt: Optimizer
    q_opt: Optimizer
    v_opt: Optimizer

    rng: PRNGKey


def _update(state: TrainState, batch: Batch, discount: float, tau: float,
            expectile: float,
            temperature: float) -> Tuple[TrainState, InfoDict]:

    key, rng = jax.random.split(state.rng)

    v_net, v_opt, v_info = update_v(state.v_opt, state.target_q_net,
                                    state.v_net, batch, expectile)

    actor, actor_opt, actor_info = awr_update_actor(key, state.actor_opt,
                                                    state.actor,
                                                    state.target_q_net, v_net,
                                                    batch, temperature)

    q_net, q_opt, q_info = update_q(state.q_opt, state.q_net, v_net, batch,
                                    discount)

    target_q_net = update_target(q_net, state.target_q_net, tau)

    new_train_state = state.replace(
        actor=actor,
        q_net=q_net,
        v_net=v_net,
        target_q_net=target_q_net,
        actor_opt=actor_opt,
        q_opt=q_opt,
        v_opt=v_opt,
        rng=rng,
    )

    info = {**q_info, **v_info, **actor_info}

    return new_train_state, info


class IQLLearner(Learner):

    def __init__(self, devices: DeviceList, rng: PRNGKey, env: gym.Env,
                 dataloader: RandomBatchDataloader, max_steps: int,
                 config: IQLConfig):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        self.config = config
        self.expectile = config.expectile
        self.tau = config.tau
        self.discount = config.discount
        self.temperature = config.temperature

        self.env = env

        rng, actor_key, q_key, v_key = jax.random.split(rng, 4)

        # dummy inputs
        observations = env.observation_space.sample()[np.newaxis]
        actions = env.action_space.sample()[np.newaxis]

        hidden_dims = tuple(config.hidden_dims)
        action_dim = actions.shape[-1]

        # Actors
        actor_def = policies.NormalTanhPolicy(hidden_dims,
                                              action_dim,
                                              log_std_scale=1e-3,
                                              log_std_min=-5.0,
                                              dropout_rate=config.dropout_rate,
                                              state_dependent_std=False,
                                              tanh_squash_distribution=False)

        if config.opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-config.actor_lr,
                                                      max_steps)
            actor_tx = optax.chain(optax.scale_by_adam(),
                                   optax.scale_by_schedule(schedule_fn))
        else:
            actor_tx = optax.adam(learning_rate=config.actor_lr)

        actor = Model.create(actor_def, actor_key, observations)
        actor_opt = Optimizer.create(actor_tx, actor)

        # q nets
        q_net_def = DoubleQNet(hidden_dims)
        q_net = Model.create(q_net_def, q_key, observations, actions)
        q_opt = Optimizer.create(optax.adam(learning_rate=config.q_lr), q_net)
        target_q_net = Model.create(q_net_def, q_key, observations, actions)

        # v nets
        v_net_def = ValueNet(hidden_dims)
        v_net = Model.create(v_net_def, v_key, observations)
        v_opt = Optimizer.create(optax.adam(learning_rate=config.v_lr), v_net)

        # train state
        super().__init__(
            devices,
            dataloader,
            TrainState,
            replicated=dict(
                actor=actor,
                q_net=q_net,
                v_net=v_net,
                target_q_net=target_q_net,
                actor_opt=actor_opt,
                q_opt=q_opt,
                v_opt=v_opt,
            ),
            rngs=dict(rng=rng, ),
            update_fn=_update,
            static_argnums=(2, 3, 4, 5),
        )

    def get_actor(self) -> Model:
        return self.train_state.actor

    def __call__(self, step: int) -> InfoDict:
        batch = self.sample_batch()
        self.train_state, infos = self.update_fn(self.train_state, batch,
                                                 self.discount, self.tau,
                                                 self.expectile,
                                                 self.temperature)
        return infos

    def save(self, log_dir: str, step: int):
        checkpoints.save_checkpoint(log_dir,
                                    self.train_state,
                                    step=step,
                                    prefix="iql_checkpoint_",
                                    keep=1,
                                    overwrite=True)
