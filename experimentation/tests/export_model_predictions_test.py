from pathlib import Path

import pandas as pd
from _pytest._py.path import LocalPath

from experimentation.export_model_predictions import export_model_predictions
from nxontology_ml.model.config import ModelConfig


def test_export_model_predictions(tmpdir: LocalPath) -> None:
    export_file = Path(tmpdir) / "precisions.tsv"
    test_conf = ModelConfig(iterations=10)
    export_model_predictions(
        export_file=export_file,
        model_config=test_conf,
        take=10,
    )
    precisions = pd.read_csv(export_file, sep="\t")
    assert precisions.shape == (10, 33)
