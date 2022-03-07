# RL framework in JAX (Flax)

Offline RL framework in JAX that supports 
- Distributed multi-device training (GPU/TPU)
- Multiprocessed dataloaders (similar to PyTorch)
- Modular experiment management and logging

Currently used by myself for research so features may change often.

### Install dependencies

```bash
conda env create -f environment.yml

conda activate rjax

# Installs dependencies for GPU, don't run if you want CPU install only
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

pip install -e .
```

To install on TPU

```bash
./gcp/setup_tpu.sh
```

There are also two Dockerfiles provided for CPU and GPU.

### Documentation

No official documentation page yet but the code is extensively documented and configs are explained.  

### Examples

IQL: [implicit_q_learning](https://github.com/ikostrikov/implicit_q_learning).

```bash
python examples/train_iql.py --prefix iql --env antmaze-medium-play-v2 --num_workers 2
```
