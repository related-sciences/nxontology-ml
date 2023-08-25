import numpy as np

from nxontology_ml.data import read_training_data


def test_read_training_data() -> None:
    x, y = read_training_data()
    assert len(x) == len(y) == 20390

    x, y = read_training_data(take=100)
    assert len(x) == len(y) == 100

    # Test order
    N = 10
    x_shuffle, _ = read_training_data(shuffle=True, take=N)
    x_unsorted, _ = read_training_data(take=N)
    assert not np.array_equiv(x_unsorted, x_shuffle)

    x_sorted, _ = read_training_data(sort=True, take=N)
    assert not np.array_equiv(x_shuffle, x_sorted)
