import logging
from collections.abc import Iterator
from typing import Any

import pandas as pd
from nxontology import NXOntology
from nxontology.node import Node as T_Node
from nxontology.node import Node_Info


class NodeInfoFeatures(Node_Info[T_Node]):
    @property
    def n_parents(self) -> int:
        return len(self.parents)

    @property
    def n_roots(self) -> int:
        return len(self.roots)

    @property
    def n_children(self) -> int:
        return len(self.children)

    @property
    def n_leaves(self) -> int:
        return len(self.leaves)

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
        "n_parents",
        "n_children",
        "n_roots",
        "n_leaves",
    ]

    def get_metrics(self) -> dict[str, Any]:
        metrics = {m: getattr(self, m) for m in self.metric_fields}
        return metrics


class NxontologyFeatures(NXOntology[T_Node]):
    @classmethod
    def _get_node_info_cls(cls) -> type[NodeInfoFeatures[T_Node]]:
        return NodeInfoFeatures

    def node_info(self, node: T_Node) -> NodeInfoFeatures[T_Node]:
        info = super().node_info(node)
        assert isinstance(info, NodeInfoFeatures)
        return info

    def get_features(self) -> Iterator[dict[str, Any]]:
        """Generate features for all nodes in an nxontology."""
        self.freeze()
        nodes = sorted(self.graph)
        for node in nodes:
            info = self.node_info(node)
            yield info.get_metrics()

    def get_features_df(self) -> pd.DataFrame:
        """Generate features for all nodes in an nxontology as a pandas DataFrame."""
        feature_df = pd.DataFrame(self.get_features())
        logging.info(
            f"Generated {len(feature_df.columns):,} features for {len(feature_df):,} nodes."
        )
        logging.info(f"Features generated: {list(feature_df.columns)}")
        return feature_df
