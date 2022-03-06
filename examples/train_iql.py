from dataclasses import dataclass

from simple_parsing import ArgumentParser

from rjax.agents.iql.learner import IQLConfig, IQLLearner
from rjax.datasets.d4rl.dataset import D4RLDataset
from rjax.datasets.dataloader import DataloaderConfig, RandomBatchDataloader
from rjax.environments.utils import make_env
from rjax.experiment.config import BaseConfig
from rjax.experiment.evaluation import EvalConfig, EvalManager
from rjax.experiment.experiment import Experiment


@dataclass
class Config(BaseConfig):
    train: IQLConfig = IQLConfig()
    eval: EvalConfig = EvalConfig()
    dataloader: DataloaderConfig = DataloaderConfig()


def main(config: Config):
    experiment = Experiment(config)

    env = make_env(config.env, config.seed)

    dataset = D4RLDataset(config.env, env)
    dataloader = RandomBatchDataloader(experiment.devices, dataset,
                                       config.dataloader)
    experiment.register_cm(dataloader)

    learner = IQLLearner(experiment.devices, experiment.split_rng(), env,
                         dataloader, config.max_steps, config.train)

    eval_manager = EvalManager(experiment.split_rng(), learner, env,
                               config.eval)

    experiment.register_callback(learner, log_freq=config.train.log_freq)
    experiment.register_callback(eval_manager, freq=config.eval.freq)

    with experiment:
        experiment.enter_registered_contexts()
        experiment.start_loop()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    main(args.config)
