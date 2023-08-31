from pathlib import Path

import pandas as pd
from nxontology import NXOntology
from scipy.spatial import distance
from sklearn.pipeline import make_pipeline

from nxontology_ml.data import read_training_data
from nxontology_ml.features import PrepareNodeFeatures
from nxontology_ml.sklearn_transformer import (
    NodeFeatures,
)
from nxontology_ml.tests.utils import read_test_dataframe
from nxontology_ml.text_embeddings.embeddings_model import (
    DEFAULT_EMBEDDING_MODEL,
    AutoModelEmbeddings,
)
from nxontology_ml.text_embeddings.text_embeddings_transformer import (
    TextEmbeddingsTransformer,
)


def _compare_vectors(df1: pd.DataFrame, df2: pd.DataFrame, tol: float = 1e-3) -> None:
    # MacOS and Linux (build) don't yield exactly the same vectors :(
    assert df1.keys().tolist() == df2.keys().tolist()
    assert all(
        distance.cosine(df1.to_numpy()[i], df2.to_numpy()[i]) < tol
        for i in range(len(df1))
    )


def test_end_to_end(nxo: NXOntology[str], embeddings_test_cache: Path) -> None:
    cached_ame = AutoModelEmbeddings.from_pretrained(
        pretrained_model_name=DEFAULT_EMBEDDING_MODEL,
        cache_path=embeddings_test_cache,
    )
    pnf = PrepareNodeFeatures(nxo)
    X, y = read_training_data(nxo=nxo, take=10, sort=True)

    # Disabled testing
    tet = TextEmbeddingsTransformer.from_config(
        enabled=False, embedding_model=cached_ame
    )
    nf = make_pipeline(pnf, tet).fit_transform(X, y)
    assert isinstance(nf, NodeFeatures)
    assert len(nf.cat_features) == 0
    assert len(nf.num_features) == 0

    # Full embedding Testing
    tet = TextEmbeddingsTransformer.from_config(
        use_lda=False, embedding_model=cached_ame
    )
    nf = make_pipeline(pnf, tet).fit_transform(X, y)
    assert isinstance(nf, NodeFeatures)
    assert len(nf.cat_features) == 0
    _compare_vectors(nf.num_features, read_test_dataframe("text_embeddings_full.json"))

    # LDA Testing
    tet = TextEmbeddingsTransformer.from_config(
        use_lda=True, embedding_model=cached_ame
    )
    nf = make_pipeline(pnf, tet).fit_transform(X, y)
    assert isinstance(nf, NodeFeatures)
    assert len(nf.cat_features) == 0
    _compare_vectors(nf.num_features, read_test_dataframe("text_embeddings_lda.json"))

    # PCA Testing
    tet = TextEmbeddingsTransformer.from_config(
        use_lda=False, pca_components=8, embedding_model=cached_ame
    )
    nf = make_pipeline(pnf, tet).fit_transform(X, y)
    assert isinstance(nf, NodeFeatures)
    assert len(nf.cat_features) == 0
    _compare_vectors(nf.num_features, read_test_dataframe("text_embeddings_pca.json"))

    # Make sure no network calls were made
    assert dict(cached_ame._counter) == {"AutoModelEmbeddings/CACHE_HIT": 50}
