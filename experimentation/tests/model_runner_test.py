from pathlib import Path

from experimentation.metadata_helpers import ModelConfig
from experimentation.model_runner import run_experiments
from nxontology_ml.text_embeddings.embeddings_model import (
    DEFAULT_EMBEDDING_MODEL,
    AutoModelEmbeddings,
)


def test_run_experiments(tmp_path: Path) -> None:
    ame = AutoModelEmbeddings.from_pretrained(
        DEFAULT_EMBEDDING_MODEL, cache_dir=tmp_path
    )
    experiments = [
        ModelConfig(
            eval_metric="BiasedMaeMetric",
            base_dir=tmp_path,
        )
    ]
    # Note about `take`: We need enough points so that both splits have over 1 class
    run_experiments(experiments, ame=ame, take=50, n_splits=2)
    # Make sure no call was made over the wire
    assert dict(ame._counter) == {}
