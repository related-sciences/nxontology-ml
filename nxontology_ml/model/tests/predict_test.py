from nxontology_ml.model.config import DEFAULT_MODEL_CONFIG
from nxontology_ml.model.predict import train_predict


def test_end_to_end() -> None:
    test_config = DEFAULT_MODEL_CONFIG
    test_config.iterations = 10

    df = train_predict(
        conf=test_config,
        train_take=100,
        predict_take=10,
    )
    assert df.shape == (10, 117)
