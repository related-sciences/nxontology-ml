from collections.abc import Iterator
from typing import Any

import pandas as pd
from nxontology import NXOntology
from nxontology.node import Node as T_Node
from nxontology.node import Node_Info


class NodeFeatures:
    def __init__(self, info: Node_Info[T_Node]) -> None:
        self.info = info

    def get_metrics(self) -> dict[str, Any]:
        metric_fields = [
            # identifiers
            "identifier",
            "name",
            # features
            "depth",
            "n_ancestors",
            "n_descendants",
            "intrinsic_ic",
            "intrinsic_ic_scaled",
            "intrinsic_ic_sanchez",
            "intrinsic_ic_sanchez_scaled",
        ]
        metrics = {m: getattr(self.info, m) for m in metric_fields}
        metrics["n_parents"] = len(self.info.parents)
        metrics["n_children"] = len(self.info.children)
        metrics["n_roots"] = len(self.info.roots)
        metrics["n_leaves"] = len(self.info.leaves)
        return metrics


def get_features(
    nxo: NXOntology[T_Node], node_features_class: type[NodeFeatures] = NodeFeatures
) -> Iterator[dict[str, int | str | None]]:
    """Generate features for all nodes in an nxontology."""
    nodes = sorted(nxo.graph)
    for node in nodes:
        info = nxo.node_info(node)
        yield node_features_class(info).get_metrics()


def get_features_df(
    nxo: NXOntology[T_Node], node_features_class: type[NodeFeatures] = NodeFeatures
) -> pd.DataFrame:
    """Generate features for all nodes in an nxontology as a pandas DataFrame."""
    return pd.DataFrame(get_features(nxo, node_features_class=node_features_class))
