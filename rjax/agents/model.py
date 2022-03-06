import os
from typing import Any, Callable

import flax.linen as nn
from flax import serialization, struct

from rjax.common import Params, PRNGKey


@struct.dataclass
class Model:
    params: Params
    model_def: nn.Module = struct.field(pytree_node=False)
    apply_fn: Callable[..., Any] = struct.field(pytree_node=False)

    @classmethod
    def create(cls, model_def: nn.Module, rng: PRNGKey, *args,
               **kwargs) -> 'Model':
        variables = model_def.init(rng, *args, **kwargs)

        _, params = variables.pop('params')

        return cls(
            params=params,
            model_def=model_def,
            apply_fn=model_def.apply,
        )

    @classmethod
    def restore(cls, model_def: nn.Module, load_path: str) -> 'Model':
        return cls(model_def=model_def, apply_fn=model_def.apply,
                   params=None).load(load_path)

    def __call__(self, *args, **kwargs):
        return self.apply_fn({"params": self.params}, *args, **kwargs)

    def apply(self, params: Params, *args, **kwargs):
        return self.apply_fn({"params": params}, *args, **kwargs)

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, "rb") as f:
            params = serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)
