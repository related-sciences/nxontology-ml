import numpy as np
from nxontology import NXOntology
from nxontology.node import NodeInfo

from nxontology_ml.data import get_efo_otar_slim
from nxontology_ml.sklearn_transformer import (
    DataFrameFnTransformer,
    NodeFeatures,
    NoFitTransformer,
)

_metric_fields = [
    # identifiers
    # "identifier",
    # "name",
    # features
    "depth",
    "n_ancestors",
    "n_descendants",
    "intrinsic_ic",
    "intrinsic_ic_scaled",
    "intrinsic_ic_sanchez",
    "intrinsic_ic_sanchez_scaled",
]

_len_fields = ["parents", "roots", "children", "leaves"]

_feature_names = _metric_fields + [f"n_{f}" for f in _len_fields]


def _node_info_features(n: NodeInfo[str]) -> np.array:
    features = [getattr(n, f) for f in _metric_fields] + [
        len(getattr(n, f)) for f in _len_fields
    ]
    return np.array(features)


class NodeInfoFeatures(DataFrameFnTransformer):
    def __init__(self) -> None:
        super().__init__(
            num_features_fn=_node_info_features,
            num_features_names=_feature_names,
        )


class PrepareNodeFeatures(NoFitTransformer[list[str], NodeFeatures]):
    def __init__(self, nxo: NXOntology[str] | None = None):
        self._nxo = nxo or get_efo_otar_slim()

    def transform(self, X: list[str], copy: bool | None = None) -> NodeFeatures:
        # X are nodes
        return NodeFeatures.from_nodes(nodes=[self._nxo.node_info(n) for n in X])
