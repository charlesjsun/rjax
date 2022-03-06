from typing import Any, Callable, Dict, Sequence, Tuple, Union

import flax
import jax.numpy as jnp
import numpy as np
from jax._src.lib import xla_client as xc
from jax.random import KeyArray

Params = flax.core.FrozenDict[str, Any]
PRNGKey = Union[KeyArray, jnp.ndarray]
Shape = Sequence[int]
InfoDict = Dict[str, Union[float, np.ndarray, jnp.ndarray]]
Device = xc.Device
DeviceList = Sequence[xc.Device]
UpdateFunction = Callable[[Any], Tuple[Any, InfoDict]]

PMAP_AXIS_NAME = "device"
