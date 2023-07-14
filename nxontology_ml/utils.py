from pathlib import Path

from nxontology import NXOntology
from nxontology.node import NodeT

ROOT_DIR: Path = Path(__file__).parent.parent


def get_output_directory(nxo: NXOntology[NodeT], parent_dir: Path = ROOT_DIR) -> Path:
    """Get output directory for an nxontology, using the ontology name for the directory."""
    assert nxo.name is not None
    directory = parent_dir.joinpath("output", nxo.name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
