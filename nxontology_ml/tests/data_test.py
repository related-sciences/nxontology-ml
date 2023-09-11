from nxontology_ml.data import read_training_data


def test_read_training_data() -> None:
    x, y = read_training_data()
    assert len(x) == len(y) == 20390

    x, y = read_training_data(take=100)
    assert len(x) == len(y) == 100
