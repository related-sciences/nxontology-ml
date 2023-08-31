from experimentation.embeddings_warmup import warmup_cache
from nxontology_ml.text_embeddings.embeddings_model import (
    DEFAULT_EMBEDDING_MODEL,
    AutoModelEmbeddings,
)
from nxontology_ml.utils import ROOT_DIR


def test_warmup_cache() -> None:
    ame = AutoModelEmbeddings.from_pretrained(
        DEFAULT_EMBEDDING_MODEL,
        cache_path=ROOT_DIR
        / "nxontology_ml/text_embeddings/tests/test_resources/embeddings_cache.ldb",
    )
    warmup_cache(ame=ame, take=10)
    assert dict(ame._counter) == {"AutoModelEmbeddings/CACHE_HIT": 10}
