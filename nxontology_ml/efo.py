from collections import Counter

import numpy as np
from nxontology.node import NodeInfo

from nxontology_ml.sklearn_transformer import DataFrameFnTransformer


def _get_curie_prefix(curie: str | None) -> str:
    """
    curie format: "{prefix}:{identifier}"
    """
    assert isinstance(curie, str)
    prefix, _ = curie.split(":", 1)
    return prefix.lower()


_efo_classification_xref_prefixes = {
    "doid": "doid",
    "gard": "gard",
    "icd10": "icd10",
    "icd9": "icd9",
    "meddra": "meddra",
    "mesh": "mesh",
    "mondo": "mondo",
    "ncit": "ncit",
    "omim": "omim",
    "omim.ps": "omimps",
    "orphanet": "orphanet",
    "snomedct": "snomedct",
    "umls": "umls",
}

_cat_feature_names = ["prefix", "is_gwas_trait"]
_num_feature_names = [
    f"xref__{name}__count" for name in _efo_classification_xref_prefixes.values()
]


def _node_xref_cat_features(n: NodeInfo[str]) -> np.array:
    return np.array([_get_curie_prefix(n.identifier), n.data.get("gwas_trait", False)])


def _node_xref_num_features(n: NodeInfo[str]) -> np.array:
    xref_counts: dict[str, int] = Counter()
    for xref in n.data.get("xrefs") or []:
        xref_counts[_get_curie_prefix(xref)] += 1
    return np.array(
        [xref_counts[prefix] for prefix in _efo_classification_xref_prefixes.keys()]
    )


class NodeXrefFeatures(DataFrameFnTransformer):
    def __init__(self) -> None:
        super().__init__(
            num_features_fn=_node_xref_num_features,
            num_features_names=_num_feature_names,
            cat_features_fn=_node_xref_cat_features,
            cat_features_names=_cat_feature_names,
        )
