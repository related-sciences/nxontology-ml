import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.pipeline import Pipeline, make_pipeline

from nxontology_ml.data import read_training_data
from nxontology_ml.efo import NodeXrefFeatures
from nxontology_ml.features import NodeInfoFeatures, PrepareNodeFeatures
from nxontology_ml.model import MODEL_SEED, CatBoostDataFormatter


def test_e2e_model() -> None:
    features_pipeline: Pipeline = make_pipeline(
        PrepareNodeFeatures(),
        NodeInfoFeatures(),
        NodeXrefFeatures(),
        CatBoostDataFormatter(),
    )
    model = CatBoostClassifier(iterations=10, silent=True, random_seed=MODEL_SEED)
    pipeline: Pipeline = make_pipeline(
        features_pipeline,
        model,
    )

    N = 100
    X, y = read_training_data(take=N + 1)
    pipeline.fit(X[:N], y[:N])
    r = pipeline.predict([X[N]])
    assert r.shape == (1, 1)

    X_t = features_pipeline.transform([X[N]])
    fi = model.get_feature_importance(data=Pool(data=X_t))
    assert fi.shape == (26,)

    fp = model.predict_proba(X=X_t)
    assert np.allclose(fp, np.array([[0.25247175, 0.22380082, 0.36932641, 0.15440102]]))

    assert model.tree_count_ == 10
