from typing import Any

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, FeaturesData, Pool
from sklearn.base import ClassifierMixin

from nxontology_ml.sklearn_transformer import NodeFeatures, NoFitTransformer


class CatBoostDataFormatter(NoFitTransformer[pd.DataFrame, catboost.FeaturesData]):
    def transform(
        self, X: NodeFeatures, copy: bool | None = None
    ) -> catboost.FeaturesData:
        return FeaturesData(
            num_feature_data=X.num_features.to_numpy(dtype=np.float32),
            cat_feature_data=X.cat_features.to_numpy(dtype=object),
            num_feature_names=list(X.num_features.columns),
            cat_feature_names=list(X.cat_features.columns),
        )


class CatBoostModel(ClassifierMixin):  # type: ignore[misc]
    def __init__(self, **kwargs: Any):
        # See CatBoostClassifier doc: https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier
        self._model: CatBoostClassifier = CatBoostClassifier(**kwargs)

    def fit(
        self, X: FeaturesData, y: list[str] | None = None, **fit_params: Any
    ) -> "CatBoostModel":
        assert y is not None, "Training data is required"
        self._model.fit(X=Pool(data=X, label=list(y)), **fit_params)
        return self

    def predict(
        self,
        X: FeaturesData,
        **predict_params: str,
    ) -> np.ndarray:
        return self._model.predict(data=Pool(data=X), **predict_params)

    def get_feature_importance(self, X: FeaturesData, **gfi_params: Any) -> Any:
        return self._model.get_feature_importance(data=Pool(data=X), **gfi_params)

    def predict_proba(self, X: FeaturesData, **predict_params: Any) -> Any:
        return self._model.predict_proba(X, **predict_params)

    @property
    def model_impl(self) -> CatBoostClassifier:
        return self._model
