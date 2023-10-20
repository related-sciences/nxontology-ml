import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tqdm import tqdm

from nxontology_ml.model.config import ModelConfig
from nxontology_ml.sklearn_transformer import (
    NodeFeatures,
)
from nxontology_ml.text_embeddings.embeddings_model import (
    DEFAULT_EMBEDDING_MODEL,
    AutoModelEmbeddings,
)

# Set global random seed for  reproducibility
np.random.seed(seed=42)


class TextEmbeddingsTransformer(TransformerMixin):  # type: ignore[misc]
    """
    Fetches text embedding for each node and by default apply dimensionality reduction on the output.
    Note: CatBoost has building dimensionality reduction, but haven't tried it yet
    """

    def __init__(
        self,
        enabled: bool,
        lda: LDA | None,
        pca: PCA | None,
        embedding_model: AutoModelEmbeddings,
        max_workers: int | None = None,
    ):
        self._enabled = enabled
        self._lda = lda
        self._pca = pca
        self._embedding_model = embedding_model
        self._max_workers = max_workers or os.cpu_count() or 1

    def fit(
        self,
        X: NodeFeatures,
        y: list[str] | None = None,
        **fit_params: NodeFeatures,
    ) -> "TextEmbeddingsTransformer":
        if not self._enabled:
            return self
        if self._lda:
            self._lda.fit(self._nodes_to_vec(X), y)
        if self._pca:
            self._pca.fit(self._nodes_to_vec(X))
        return self

    def transform(self, X: NodeFeatures, copy: bool | None = None) -> NodeFeatures:
        if not self._enabled:
            return X
        text_embeddings: dict[str, np.ndarray] = {"te": self._nodes_to_vec(X)}
        if self._lda:
            text_embeddings["lda_te"] = self._lda.transform(text_embeddings["te"])
        if self._pca:
            text_embeddings["pca_te"] = self._pca.transform(text_embeddings["te"])
        if self._lda or self._pca:
            # If some dimensionality reduction technique is used, we discard the full length vectors
            del text_embeddings["te"]

        named_text_embeddings = []
        keys = list(text_embeddings.keys())
        for zipped_vecs in zip(*text_embeddings.values(), strict=True):
            named_text_embeddings.append(
                {
                    f"{prefix}_{i}": x
                    for prefix, vec in zip(keys, zipped_vecs, strict=True)
                    for i, x in enumerate(vec)
                }
            )

        X.num_features = pd.concat(
            [X.num_features, pd.DataFrame(named_text_embeddings)], axis=1
        )
        return X

    def _nodes_to_vec(self, X: NodeFeatures) -> np.ndarray:
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            parallel_it = executor.map(self._embedding_model.embed_node, X.nodes)
            return np.array(
                list(
                    tqdm(
                        parallel_it,
                        desc="Fetching node embeddings",
                        total=len(X.nodes),
                        delay=5,
                    )
                )
            )

    @classmethod
    def from_config(
        cls,
        conf: ModelConfig,
        pretrained_model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedding_model: AutoModelEmbeddings | None = None,
    ) -> "TextEmbeddingsTransformer":
        return cls(
            enabled=conf.embedding_enabled,
            lda=LDA() if conf.use_lda else None,
            pca=PCA(n_components=conf.pca_components) if conf.pca_components else None,
            embedding_model=embedding_model
            or AutoModelEmbeddings.from_pretrained(
                pretrained_model_name=pretrained_model_name,
                cache_dir=conf.cache_dir,
            ),
        )
