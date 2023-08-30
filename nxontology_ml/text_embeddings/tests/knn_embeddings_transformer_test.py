import re
from pathlib import Path

import pytest
from nxontology import NXOntology
from sklearn.pipeline import make_pipeline

from nxontology_ml.data import read_training_data
from nxontology_ml.features import PrepareNodeFeatures
from nxontology_ml.sklearn_transformer import NodeFeatures
from nxontology_ml.tests.utils import assert_frame_equal_to
from nxontology_ml.text_embeddings.embeddings_model import (
    DEFAULT_EMBEDDING_MODEL,
    AutoModelEmbeddings,
)
from nxontology_ml.text_embeddings.faiss_helpers import _batched
from nxontology_ml.text_embeddings.knn_embeddings_transformer import (
    KnnEmbeddingsTransformer,
)


def test_end_to_end(nxo: NXOntology[str], embeddings_test_cache: Path) -> None:
    cached_ame = AutoModelEmbeddings.from_pretrained(
        pretrained_model_name=DEFAULT_EMBEDDING_MODEL,
        cache_path=embeddings_test_cache,
    )
    pnf = PrepareNodeFeatures(nxo)
    X, y = read_training_data(nxo=nxo, take=10, sort=True)

    ##
    # Main test
    ket = KnnEmbeddingsTransformer.from_config(
        k_neighbors=10,
        index_batch_size=2,
        embedding_model=cached_ame,
    )
    nf = make_pipeline(pnf, ket).fit_transform(X, y)
    assert isinstance(nf, NodeFeatures)
    assert len(nf.cat_features) == 0
    assert_frame_equal_to(nf.num_features, "text_embeddings_knn.json")

    ##
    # Disabled test
    ket = KnnEmbeddingsTransformer.from_config(
        enabled=False,
        embedding_model=cached_ame,
    )
    nf = make_pipeline(pnf, ket).fit_transform(X, y)
    assert isinstance(nf, NodeFeatures)
    assert len(nf.cat_features) == 0
    assert len(nf.num_features) == 0

    ##
    # Make sure no network calls were made
    assert dict(cached_ame._counter) == {"AutoModelEmbeddings/CACHE_HIT": 20}


def test_batched() -> None:
    b = list(_batched(range(5), n=2))
    assert b == [(0, 1), (2, 3), (4,)]

    with pytest.raises(ValueError, match=re.escape("n must be at least one")):
        list(_batched(range(5), n=0))
