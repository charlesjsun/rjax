import logging
import multiprocessing
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import List

from simple_parsing import choice

from rjax.common import DeviceList
from rjax.datasets.dataset import Dataset


@dataclass
class DataloaderConfig:
    # size of a single batch (should be divisible by number of devices)
    batch_size: int = 256
    # number of worker processes to spawn. 0 means no multiprocessing
    num_workers: int = 0
    # how many batches should be prefetched in the queue
    num_prefetch: int = 16
    # worker method: fork or spawn. fork is typically faster and less memory intensive since it uses
    # copy-on-write so it won't copy your whole dataset for each worker. However, be sure to
    # not access too many outside references inside your Dataset or change your dataset,
    # otherwise memory will become an issue.
    # Use spawn if you plan on writing lazy-loaded dataloader that reads from disk and don't store a lot
    # of dynamic memory.
    worker_method: str = choice("fork", "spawn", default="fork")


def _random_batch_worker_fn(dataset: Dataset, config: DataloaderConfig,
                            input_queue: multiprocessing.Queue,
                            output_queue: multiprocessing.Queue):
    while True:
        # wait until input
        _ = input_queue.get()
        batch = dataset.get_random_batch(config.batch_size)
        output_queue.put(batch)


class RandomBatchDataloader(AbstractContextManager):
    """ Dataloader that returns a random batch. """

    def __init__(self, devices: DeviceList, dataset: Dataset,
                 config: DataloaderConfig):
        self.dataset = dataset
        self.config = config
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        assert self.batch_size % len(devices) == 0, (
            "batch_size must be divisible by number of devices")

    def sample(self):
        if self.num_workers <= 0:
            return self.dataset.get_random_batch(self.batch_size)

        self.input_queue.put(1)
        batch = self.output_queue.get()
        return batch

    def __enter__(self):
        if self.num_workers <= 0:
            return self

        ctx = multiprocessing.get_context(self.config.worker_method)
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()

        self.workers: List[multiprocessing.Process] = []
        for i in range(self.num_workers):
            worker = ctx.Process(target=_random_batch_worker_fn,
                                 args=(self.dataset, self.config,
                                       self.input_queue, self.output_queue),
                                 daemon=True)
            self.workers.append(worker)
            worker.start()
            logging.info(f"Started RandomBatchDataloader worker[{i}]")

        # prefetch
        for _ in range(self.config.num_prefetch):
            self.input_queue.put(1)

        return self

    def __exit__(self, *args, **kwargs):
        if self.num_workers <= 0:
            return

        self.output_queue.close()
        self.input_queue.close()

        for i, worker in enumerate(self.workers):
            if worker.is_alive():
                worker.terminate()
                worker.join()
            worker.close()
            logging.info(f"Stopped RandomBatchDataloader worker[{i}]")
