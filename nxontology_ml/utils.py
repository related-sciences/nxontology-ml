from pathlib import Path

from nxontology import NXOntology
from nxontology.node import Node as T_Node


def get_root_directory() -> Path:
    """Get root directory for the project."""
    return Path(__file__).parent.parent


def get_output_directory(nxo: NXOntology[T_Node]) -> Path:
    """Get output directory for an nxontology, using the ontology name for the directory."""
    assert nxo.name is not None
    directory = get_root_directory().joinpath("output", nxo.name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
