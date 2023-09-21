from pathlib import Path

import pandas as pd
from _pytest._py.path import LocalPath

from experimentation.export_model_predictions import export_model_predictions
from nxontology_ml.text_embeddings.embeddings_model import (
    DEFAULT_EMBEDDING_MODEL,
    AutoModelEmbeddings,
)


def test_export_model_predictions(tmpdir: LocalPath) -> None:
    export_file = Path(tmpdir) / "precisions.tsv"
    AutoModelEmbeddings.from_pretrained(DEFAULT_EMBEDDING_MODEL)
    export_model_predictions(
        export_file=export_file,
        take=10,
        text_embeddings_enabled=False,
    )
    precisions = pd.read_csv(export_file, sep="\t")
    assert precisions.shape == (10, 33)
