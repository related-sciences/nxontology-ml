import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline

from nxontology_ml.data import read_training_data
from nxontology_ml.efo import NodeXrefFeatures
from nxontology_ml.features import NodeInfoFeatures, PrepareNodeFeatures
from nxontology_ml.model import CatBoostDataFormatter, CatBoostModel


def test_e2e_model() -> None:
    features_pipeline: Pipeline = make_pipeline(
        PrepareNodeFeatures(),
        NodeInfoFeatures(),
        NodeXrefFeatures(),
        CatBoostDataFormatter(),
    )
    model = CatBoostModel(iterations=10, silent=True, random_seed=42)
    pipeline: Pipeline = make_pipeline(
        features_pipeline,
        model,
    )

    N = 100
    X, y = read_training_data(sort=True, take=N + 1)
    pipeline.fit(X[:N], y[:N])
    r = pipeline.predict([X[N]])
    assert r.shape == (1, 1)
    assert r[0][0] == "02-disease-root"

    X_t = features_pipeline.transform([X[N]])
    fi = model.get_feature_importance(X=X_t)
    assert fi.shape == (26,)

    fp = model.predict_proba(X=X_t)
    assert np.allclose(fp, np.array([[0.23435558, 0.63125101, 0.07538654, 0.05900687]]))

    assert model.model_impl.tree_count_ == 10
