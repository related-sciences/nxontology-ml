import logging
from collections.abc import Callable, Mapping
from typing import Any

from fire import Fire

from nxontology_ml.efo import write_efo_features


def cli(fire_fn: Callable[[Mapping[str, Any]], None] = Fire) -> None:
    """
    Run like `poetry run nxontology_ml`
    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    commands = {
        "efo": write_efo_features,
    }
    fire_fn(commands)
