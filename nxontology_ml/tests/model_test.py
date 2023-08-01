from nxontology_ml.model import read_training_data, to_features_data, train_model


def test_end_to_end() -> None:
    # train on 100 points, make sure nothing crashes
    train_model(to_features_data(read_training_data(take=100)))
