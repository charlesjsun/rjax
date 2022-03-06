import collections

import gym
import numpy as np


def clip_actions_to_eps(eps=1e-5):

    def inner(dataset):
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)
        return dataset

    return inner


def offset_rewards(offset=-1.0):

    def inner(dataset):
        dataset["rewards"] = dataset["rewards"] + offset
        return dataset

    return inner


def normalize():

    def inner(dataset):
        return dataset

    return inner


def compose(*fns):

    def inner(dataset):
        for f in fns:
            dataset = f(dataset)
        return dataset

    return inner


def get_preprocessing_fn(env_name):
    if "antmaze" in env_name:
        return compose(
            clip_actions_to_eps(1e-5),
            offset_rewards(-1.0),
        )
    elif ("halfcheetah" in env_name or "walker2d" in env_name
          or 'hopper' in env_name):
        return compose(
            clip_actions_to_eps(1e-5),
            normalize(),
        )
    else:
        return lambda dataset: dataset


def sequence_dataset_iter(env: gym.Env, dataset):
    dataset_keys = {
        'observations', 'actions', 'rewards', 'terminals', 'timeouts'
    }

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    step_ = 0
    next_obs_ = dataset["observations"][0]
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (step_ == env._max_episode_steps - 1)

        # skip if trajectory timed out
        if done_bool or not final_timestep:
            step_ += 1
            for k in dataset_keys:
                data_[k].append(dataset[k][i])
            next_obs_ = dataset["observations"][(i + 1) % N]

        if done_bool or final_timestep:
            if step_ == 0:
                continue
            episode_step = step_
            episode_data = {}
            data_["observations"].append(next_obs_)
            for k in data_:
                episode_data[k] = np.array(data_[k])
            episode_data["steps"] = episode_step
            yield episode_data
            data_ = collections.defaultdict(list)
            step_ = 0
