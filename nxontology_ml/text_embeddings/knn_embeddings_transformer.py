from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from nxontology.node import NodeInfo
from sklearn.base import TransformerMixin

from nxontology_ml.sklearn_transformer import (
    NodeFeatures,
)
from nxontology_ml.text_embeddings.embeddings_model import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_SIZES,
    AutoModelEmbeddings,
)
from nxontology_ml.text_embeddings.faiss_helpers import (
    DEFAULT_FAISS_INDEX,
    FaissIndexBuilder,
    LabeledNodeVec,
    NodeIdLabel,
)


class KnnEmbeddingsTransformer(TransformerMixin):  # type: ignore[misc]
    def __init__(
        self,
        enabled: bool,
        index_builder: FaissIndexBuilder,
        embedding_model: AutoModelEmbeddings,
        k_neighbors: int,
        index_batch_size: int,
        identical_dist_threshold: float = 0.0001,
    ):
        self._enabled = enabled
        self._index_builder = index_builder
        self._embedding_model = embedding_model
        self._k_neighbors = k_neighbors
        self._index_batch_size = index_batch_size
        self._identical_dist_threshold = identical_dist_threshold

    def fit(
        self,
        X: NodeFeatures,
        y: list[str] | None = None,
        **fit_params: Any,
    ) -> "KnnEmbeddingsTransformer":
        if not self._enabled:
            return self
        assert y is not None, "This transformer needs labels"
        self._index_builder.add_nodes(
            node_vecs=(
                self._labeled_node_to_vec(node, label)
                for node, label in zip(X.nodes, y, strict=True)
            ),
            batch_size=self._index_batch_size,
        )
        return self

    def transform(self, X: NodeFeatures, copy: bool | None = None) -> NodeFeatures:
        """
        For each target node:
          - Calculate its text embedding (based on node name & definition)
          - Lookup K labelled nearest neighbors (from training set)
          - Group the NNS by label
          - Calculate descriptive statistics (min, max, median, q1, q3 & support) for each label
        """
        if not self._enabled:
            return X
        index = self._index_builder.build_index()
        assert (
            index.ntotal > 0
        ), 'Transformer requires to be "fitted" before calling ".transform"'

        node_embeddings = []
        for node in X.nodes:
            vec = self._embedding_model.embed_node(node)
            dists_by_labels = defaultdict(list)
            cnt = 0
            for nn in index.search(vec, k=self._k_neighbors + 1):
                # Because we index the training nodes, we need to remove each target when fetching neighbors
                if nn.dist > self._identical_dist_threshold and cnt < self._k_neighbors:
                    cnt += 1
                    dists_by_labels[nn.node_id_label.label].append(nn.dist)
            node_features = {}
            for lbl, short_name in self._lbl_short_names.items():
                dists = dists_by_labels[lbl]
                node_features[f"{short_name}-support"] = len(dists)
                (
                    node_features[f"{short_name}-min"],
                    node_features[f"{short_name}-q.5"],
                    node_features[f"{short_name}-q1"],
                    node_features[f"{short_name}-med"],
                    node_features[f"{short_name}-max"],
                ) = (
                    np.quantile(dists, q=self._q) if len(dists) > 0 else self._all_nans
                )
            node_embeddings.append(node_features)

        X.num_features = pd.concat(
            [X.num_features, pd.DataFrame(node_embeddings)], axis=1
        )
        return X

    _lbl_short_names = {
        "01-disease-subtype": "L1",
        "02-disease-root": "L2",
        "03-disease-area": "L3",
        "04-non-disease": "L4",
    }
    # Quantiles
    _q = [0.0, 0.125, 0.25, 0.5, 1.0]
    _all_nans = [np.nan] * len(_q)

    def _labeled_node_to_vec(self, node: NodeInfo[str], label: str) -> LabeledNodeVec:
        node_id = node.identifier
        assert node_id
        return LabeledNodeVec(
            node_id_label=NodeIdLabel(node_id=node_id, label=label),
            vec=self._embedding_model.embed_node(node),
        )

    @classmethod
    def from_config(
        cls,
        enabled: bool = True,
        pretrained_model_name: str = DEFAULT_EMBEDDING_MODEL,
        nns_index_name: str = DEFAULT_FAISS_INDEX,
        k_neighbors: int = 50,
        index_batch_size: int = 32,
        embedding_model: AutoModelEmbeddings | None = None,
    ) -> "KnnEmbeddingsTransformer":
        return cls(
            enabled=enabled,
            index_builder=FaissIndexBuilder.from_index_name(
                index_name=nns_index_name, dim=EMBEDDING_SIZES[pretrained_model_name]
            ),
            embedding_model=embedding_model
            or AutoModelEmbeddings.from_pretrained(
                pretrained_model_name=pretrained_model_name
            ),
            k_neighbors=k_neighbors,
            index_batch_size=index_batch_size,
        )
