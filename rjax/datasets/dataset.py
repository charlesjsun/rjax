from typing import Tuple, Union

import numpy as np


class Dataset:
    """ A base Dataset class. One of the two below methods must be implemented to use their respective Dataloader. """

    @property
    def size(self) -> int:
        """ Size of dataset, must be implemented if __getitem__ was to be used with Dataloader. """
        raise NotImplementedError

    def __getitem__(self,
                    idx: int) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """ Get the idx element of the dataset. """
        raise NotImplementedError

    def get_random_batch(
            self,
            batch_size: int) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """ Get a random batch of batch_size. """
        raise NotImplementedError
