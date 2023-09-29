import re

import pytest

from experimentation.metadata_helpers import (
    ModelConfig,
    ModelMetadataBuilder,
    df_from_all_experiments,
)
from nxontology_ml.tests.utils import get_test_resource_path


def test_df_from_all_experiments() -> None:
    # To create this model, the test in `model_runner_test.py` was run and the output was saved
    df_from_all_experiments(get_test_resource_path("20230831_195414_mae"))


def test_experiment_name() -> None:
    experiment = ModelConfig(
        eval_metric="BiasedMaeMetric",
        depth=7,
        embedding_enabled=True,
        pca_components=64,
        use_lda=True,
        use_knn=True,
    )
    assert experiment.name == "pca64_lda_knn_d7_mae"

    experiment = ModelConfig(
        eval_metric="BiasedMaeMetric",
        embedding_enabled=True,
    )
    assert experiment.name == "full_embedding_mae"

    experiment = ModelConfig(
        name_override="test_name_override",
        description_override="test_description_override",
    )
    assert experiment.name == "test_name_override"
    assert experiment.description == "test_description_override"
    assert experiment.get_eval_metric == "MultiClass"


def test_experiment_already_exists() -> None:
    experiment = ModelConfig(
        eval_metric="BiasedMaeMetric",
        base_dir=get_test_resource_path(""),
    )
    with pytest.raises(
        ValueError,
        match=re.escape("An experiment `20230831_195414_mae` already exists"),
    ):
        ModelMetadataBuilder(experiment).get_model_dir()
