from typing import Tuple

import jax
import jax.numpy as jnp

from rjax.agents.model import Model
from rjax.agents.optimizer import Optimizer
from rjax.common import InfoDict, Params, PRNGKey
from rjax.datasets.d4rl.dataset import Batch


def update(key: PRNGKey, actor_opt: Optimizer, actor: Model, q_net: Model,
           v_net: Model, batch: Batch,
           temperature: float) -> Tuple[Model, Optimizer, InfoDict]:
    v = v_net(batch.observations)

    q1, q2 = q_net(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)
    exp_a = jnp.exp((q - v) * temperature)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply(actor_params,
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss, {'actor_loss': actor_loss, 'adv': q - v}

    return actor_opt.apply_gradient(actor_loss_fn, actor)
