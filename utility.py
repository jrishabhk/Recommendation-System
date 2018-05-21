# -*- coding: utf-8 -*-


import numbers

import numpy as np

def get_rng(random_state):
    if random_state is None:
        return np.random.mtrand._rand
    elif isinstance(random_state, (numbers.Integral, np.integer)):
        return np.random.RandomState(random_state)
    if isinstance(random_state, np.random.RandomState):
        return random_state
    raise ValueError('Wrong random state. Expecting None, an int or a numpy '
                     'RandomState instance, got a '
                     '{}'.format(type(random_state)))


