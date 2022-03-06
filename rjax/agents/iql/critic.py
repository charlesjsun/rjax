from typing import Tuple

import jax
import jax.numpy as jnp

from rjax.agents.model import Model
from rjax.agents.optimizer import Optimizer
from rjax.common import InfoDict, Params
from rjax.datasets.d4rl.dataset import Batch


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def update_v(v_opt: Optimizer, q_net: Model, v_net: Model, batch: Batch,
             expectile: float) -> Tuple[Model, Optimizer, InfoDict]:
    actions = batch.actions
    q1, q2 = q_net(batch.observations, actions)
    q = jnp.minimum(q1, q2)

    def value_loss_fn(v_net_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = v_net.apply(v_net_params, batch.observations)
        value_loss = loss(q - v, expectile).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
        }

    return v_opt.apply_gradient(value_loss_fn, v_net)


def update_q(q_opt: Optimizer, q_net: Model, target_v_net: Model, batch: Batch,
             discount: float) -> Tuple[Model, Optimizer, InfoDict]:
    next_v = target_v_net(batch.next_observations)

    target_q = batch.rewards + discount * (1.0 - batch.terminals) * next_v

    def q_loss_fn(q_net_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = q_net.apply(q_net_params, batch.observations, batch.actions)
        q_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return q_loss, {'q_loss': q_loss, 'q1': q1.mean(), 'q2': q2.mean()}

    return q_opt.apply_gradient(q_loss_fn, q_net)


def update_target(q_net: Model, target_q_net: Model, tau: float) -> Model:
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), q_net.params,
        target_q_net.params)

    return target_q_net.replace(params=new_target_params)
