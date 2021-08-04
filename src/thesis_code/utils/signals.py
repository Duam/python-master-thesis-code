import numpy as np


def rectangle(sample: int, samples_per_period: int):
    """Creates a rectangle signal.
    :param sample: The sample index
    :param samples_per_period:
    :returns: Signal value at the given sample.
    """
    return 1. if np.mod(sample, samples_per_period) < samples_per_period/2. else 0.


def triangle(sample: int, samples_per_period: int):
    """Creates a triangle signal.
    :param sample: The sample index
    :param samples_per_period:
    :returns: Signal value at the given sample.
    """
    kmodn = np.mod(sample, samples_per_period)
    is_rising = kmodn < samples_per_period/2.
    rising_val = 2*kmodn/samples_per_period
    falling_val = 2-2*kmodn/samples_per_period
    return rising_val if is_rising else falling_val
