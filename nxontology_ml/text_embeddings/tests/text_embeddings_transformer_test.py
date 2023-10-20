from pathlib import Path

from nxontology import NXOntology
from pandas._testing import assert_frame_equal
from sklearn.pipeline import make_pipeline

from nxontology_ml.data import read_training_data
from nxontology_ml.features import PrepareNodeFeatures
from nxontology_ml.model.config import ModelConfig
from nxontology_ml.sklearn_transformer import (
    NodeFeatures,
)
from nxontology_ml.tests.utils import assert_frame_equal_to
from nxontology_ml.text_embeddings.embeddings_model import (
    DEFAULT_EMBEDDING_MODEL,
    AutoModelEmbeddings,
)
from nxontology_ml.text_embeddings.text_embeddings_transformer import (
    TextEmbeddingsTransformer,
)


def test_end_to_end(
    nxo: NXOntology[str],
    embeddings_cache_dir: Path,
) -> None:
    cached_ame = AutoModelEmbeddings.from_pretrained(
        pretrained_model_name=DEFAULT_EMBEDDING_MODEL,
        cache_dir=embeddings_cache_dir,
    )
    pnf = PrepareNodeFeatures(nxo)
    X, y = read_training_data(nxo=nxo, take=10)
    conf = ModelConfig(cache_dir=embeddings_cache_dir)

    ##
    # Disabled testing
    tet = TextEmbeddingsTransformer.from_config(conf=conf, embedding_model=cached_ame)
    nf = make_pipeline(pnf, tet).fit_transform(X, y)
    assert isinstance(nf, NodeFeatures)
    assert len(nf.cat_features) == 0
    assert len(nf.num_features) == 0

    ##
    # Full embedding Testing
    conf.embedding_enabled = True
    tet = TextEmbeddingsTransformer.from_config(conf, embedding_model=cached_ame)
    nf = make_pipeline(pnf, tet).fit_transform(X, y)
    assert isinstance(nf, NodeFeatures)
    assert len(nf.cat_features) == 0
    assert_frame_equal_to(nf.num_features, "text_embeddings_full.json")
    # Determinism
    nf2 = make_pipeline(pnf, tet).fit_transform(X, y)
    assert isinstance(nf2, NodeFeatures)
    assert_frame_equal(nf.num_features, nf2.num_features)

    ##
    # LDA Testing
    conf.use_lda = True
    tet = TextEmbeddingsTransformer.from_config(conf=conf, embedding_model=cached_ame)
    nf = make_pipeline(pnf, tet).fit_transform(X, y)
    assert isinstance(nf, NodeFeatures)
    assert len(nf.cat_features) == 0
    assert nf.num_features.shape == (10, 2)
    # Determinism
    nf2 = make_pipeline(pnf, tet).fit_transform(X, y)
    assert isinstance(nf2, NodeFeatures)
    assert_frame_equal(nf.num_features, nf2.num_features)

    ##
    # PCA Testing
    conf.use_lda = False
    conf.pca_components = 8
    tet = TextEmbeddingsTransformer.from_config(conf, embedding_model=cached_ame)
    nf = make_pipeline(pnf, tet).fit_transform(X, y)
    assert isinstance(nf, NodeFeatures)
    assert len(nf.cat_features) == 0
    assert nf.num_features.shape == (10, 8)
    # Determinism
    nf2 = make_pipeline(pnf, tet).fit_transform(X, y)
    assert isinstance(nf2, NodeFeatures)
    assert_frame_equal(nf.num_features, nf2.num_features)

    # Make sure no network calls were made
    assert set(cached_ame._counter.keys()) == {"AutoModelEmbeddings/CACHE_HIT"}
