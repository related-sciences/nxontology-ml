import logging

import fire

from nxontology_ml.efo import write_efo_features


def cli() -> None:
    """
    Run like `poetry run nxontology_ml`
    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    commands = {
        "efo": write_efo_features,
    }
    fire.Fire(commands)
