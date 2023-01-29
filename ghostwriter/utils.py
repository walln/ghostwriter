import random

import numpy as np
import torch


def set_seed_and_backends(seed: int = 1337):
    """
    Sets up general utilities and seeds random number generators

    Parameters
    ----------
    seed
        The random number seed
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
