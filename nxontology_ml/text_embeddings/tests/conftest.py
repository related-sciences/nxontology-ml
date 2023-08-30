from pathlib import Path

import numpy as np
import pytest
from nxontology import NXOntology

from nxontology_ml.data import get_efo_otar_slim
from nxontology_ml.tests.utils import get_test_resource_path

# Set global random seed for test reproducibility
np.random.seed(seed=42)


@pytest.fixture
def embeddings_test_cache() -> Path:
    # We don't want to fetch embeddings over the internet during unit tests
    return get_test_resource_path("embeddings_cache.ldb")


@pytest.fixture
def nxo() -> NXOntology[str]:
    return get_efo_otar_slim()
