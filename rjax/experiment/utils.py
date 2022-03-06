import datetime
import os
from typing import Callable, Dict, Iterable, Sequence

import git

from rjax.experiment.config import BaseConfig

PROJECT_PATH = os.path.dirname(
    os.path.realpath(os.path.join(__file__, '..', '..')))


def datetimestamp() -> str:
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%dT%H-%M-%S")


def datestamp() -> str:
    return datetime.date.today().isoformat()


def timestamp() -> str:
    now = datetime.datetime.now()
    time_now = datetime.datetime.time(now)
    return time_now.strftime(f"%H-%M-%S")


def _dictdiff(new: Dict, old: Dict) -> Dict:
    d = {}
    for k in new:
        if k not in old:
            d[k] = new[k]
        else:
            if isinstance(new[k], dict) and isinstance(old[k], dict):
                diff = _dictdiff(new[k], old[k])
                if len(diff) > 0:
                    d[k] = diff
            else:
                if new[k] != old[k]:
                    d[k] = new[k]
    return d


def _flatten_dict(d: Dict, prefix: str = "") -> Dict:
    nd = {}
    for k in d:
        nk = k if prefix == "" else f"{prefix}.{k}"
        if isinstance(d[k], dict):
            nd.update(_flatten_dict(d[k], prefix=nk))
        else:
            nd[nk] = d[k]
    return nd


def _config_diff_str(config: BaseConfig, ignore: Iterable[str]) -> str:
    config_dict = config.to_dict()
    default_dict = type(config)().to_dict()

    diff_dict = _dictdiff(config_dict, default_dict)
    for k in ignore:
        if k in diff_dict:
            diff_dict.pop(k)
    diff_dict = _flatten_dict(diff_dict)

    diff_str = "--".join(
        map(lambda k: f"{k}_{diff_dict[k]}", sorted(diff_dict)))

    if diff_str == "":
        return "default"
    return diff_str


def create_log_dir_path(config: BaseConfig) -> str:
    config_ignore = set(
        {"logging", "logdir_structure", "wandb", "tb", "checkpoint"})

    # we need to populate the above set so we return lambdas instead of strs
    def parse_token(token: str) -> Callable[[], str]:
        if token == "":
            return lambda: ""
        elif token == "base":
            config_ignore.add("base")
            return lambda: config.base
        elif token == "env":
            config_ignore.add("env")
            return lambda: config.env
        elif token == "prefix":
            config_ignore.add("prefix")
            return lambda: config.prefix
        elif token == "name":
            config_ignore.add("name")
            return lambda: config.name
        elif token == "seed":
            config_ignore.add("seed")
            return lambda: str(config.seed)
        elif token == "time":
            config_ignore.add("time")
            return lambda: datetimestamp()
        elif token == "config":
            config_ignore.add("config")
            return lambda: _config_diff_str(config, config_ignore)
        else:
            raise ValueError(f"Invalid logdir token {token}")

    def parse_dir(dir: str) -> Sequence[Callable[[], str]]:
        tokens = dir.split("-")
        return [parse_token(token) for token in tokens]

    dirs = config.logdir_structure.split("/")
    dirs = [parse_dir(d) for d in dirs if d != ""]
    dirs = ["--".join(f() for f in d) for d in dirs]

    path = os.path.join(*dirs)
    return path


def get_repo(path: str = PROJECT_PATH, search_parent_directories: bool = True):
    repo = git.Repo(path, search_parent_directories=search_parent_directories)
    return repo


def get_git_rev(*args, **kwargs):
    try:
        repo = get_repo(*args, **kwargs)
        if repo.head.is_detached:
            git_rev = repo.head.object.name_rev
        else:
            git_rev = repo.active_branch.commit.name_rev
    except Exception:
        git_rev = None

    return git_rev


def git_diff(*args, **kwargs):
    repo = get_repo(*args, **kwargs)
    diff = repo.git.diff()
    return diff


def save_git_diff(savepath: str, *args, **kwargs):
    diff = git_diff(*args, **kwargs)
    with open(savepath, "w") as f:
        f.write(diff)
