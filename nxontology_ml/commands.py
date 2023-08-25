import logging
from collections.abc import Callable, Mapping
from typing import Any

from fire import Fire


def cli(fire_fn: Callable[[Mapping[str, Any]], None] = Fire) -> None:
    """
    Run like `poetry run nxontology_ml`
    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    fire_fn({})
