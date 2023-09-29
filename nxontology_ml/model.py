import catboost
import numpy as np
import pandas as pd
from catboost import FeaturesData

from nxontology_ml.sklearn_transformer import NodeFeatures, NoFitTransformer

MODEL_SEED = 42


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
