from collections.abc import Iterator

import pandas as pd
from nxontology import NXOntology
from nxontology.node import Node as T_Node
from nxontology.node import Node_Info


class NodeFeatures:
    def __init__(self, info: Node_Info[T_Node]) -> None:
        self.info = info

    def get_metrics(self) -> dict[str, int | str | None]:
        metrics = [
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
        return {m: getattr(self.info, m) for m in metrics}


def get_features(nxo: NXOntology[T_Node]) -> Iterator[dict[str, int | str | None]]:
    """Generate features for all nodes in an nxontology."""
    nodes = sorted(nxo.graph)
    for node in nodes:
        info = nxo.node_info(node)
        yield NodeFeatures(info).get_metrics()


def get_features_df(nxo: NXOntology[T_Node]) -> pd.DataFrame:
    """Generate features for all nodes in an nxontology as a pandas DataFrame."""
    return pd.DataFrame(get_features(nxo))
