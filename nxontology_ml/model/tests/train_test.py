import numpy as np

from nxontology_ml.data import get_efo_otar_slim
from nxontology_ml.model.config import DEFAULT_MODEL_CONFIG
from nxontology_ml.model.train import train_model


def test_train_determinism() -> None:
    test_config = DEFAULT_MODEL_CONFIG
    test_config.iterations = 10

    nxo = get_efo_otar_slim()
    feature_pipeline, model = train_model(
        conf=test_config,
        nxo=nxo,
        take=100,
    )

    # Feature determinism
    X_test = sorted(nxo.graph)[:50]
    X_features1 = feature_pipeline.transform(X_test)
    X_features2 = feature_pipeline.transform(X_test)
    assert np.array_equal(X_features1.cat_feature_data, X_features2.cat_feature_data)
    assert np.array_equal(X_features1.num_feature_data, X_features2.num_feature_data)

    # Model determinism
    y_proba1 = model.predict_proba(X_features1)
    y_proba2 = model.predict_proba(X_features1)
    assert np.array_equal(y_proba1, y_proba2)
