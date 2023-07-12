from pathlib import Path

from nxontology import NXOntology
from nxontology.node import NodeT


def get_root_directory() -> Path:
    """Get root directory for the project."""
    return Path(__file__).parent.parent


def get_output_directory(nxo: NXOntology[NodeT]) -> Path:
    """Get output directory for an nxontology, using the ontology name for the directory."""
    assert nxo.name is not None
    directory = get_root_directory().joinpath("output", nxo.name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
