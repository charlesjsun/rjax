import os
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import serialization, struct

from rjax.agents.model import Model
from rjax.common import PMAP_AXIS_NAME, InfoDict, Params


@struct.dataclass
class Optimizer:
    tx_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    pmapped: bool = struct.field(pytree_node=False, default=True)

    @classmethod
    def create(cls, tx: optax.GradientTransformation,
               model: Model) -> 'Optimizer':
        tx_state = tx.init(model.params)
        return cls(tx_state=tx_state, tx=tx)

    @classmethod
    def restore(cls, tx: optax.GradientTransformation,
                load_path: str) -> 'Optimizer':
        return cls(tx_state=None, tx=tx).load(load_path)

    def apply_gradient(self, loss_fn: Callable[[Params], Tuple[jnp.ndarray,
                                                               InfoDict]],
                       model: Model) -> Tuple[Model, 'Optimizer', InfoDict]:
        # compute gradient
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(model.params)

        # synchronize grad across devices
        if self.pmapped:
            grads = jax.lax.pmean(grads, axis_name=PMAP_AXIS_NAME)

        # perform gradient step
        updates, new_tx_state = self.tx.update(grads, self.tx_state,
                                               model.params)
        new_params = optax.apply_updates(model.params, updates)

        # create new model and opt
        new_model = model.replace(params=new_params)
        new_opt = self.replace(tx_state=new_tx_state)
        return new_model, new_opt, info

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(serialization.to_bytes(self.tx_state))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            tx_state = serialization.from_bytes(self.tx_state, f.read())
        return self.replace(tx_state=tx_state)
