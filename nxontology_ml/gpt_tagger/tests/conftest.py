from pathlib import Path

import pytest

from nxontology_ml.gpt_tagger import TaskConfig
from nxontology_ml.tests.utils import get_test_resource_path


@pytest.fixture
def precision_config(tmp_path: Path) -> TaskConfig:
    return TaskConfig(
        name="precision",
        prompt_path=get_test_resource_path("precision_v1.txt"),
        node_attributes=["efo_id", "efo_label", "efo_definition"],
        openai_model_name="gpt-3.5-turbo",
        model_temperature=0,
        allowed_labels=frozenset({"low", "medium", "high"}),
        logs_path=None,  # Don't log during tests (unless integration)
        cache_dir=tmp_path / "cache",
    )
