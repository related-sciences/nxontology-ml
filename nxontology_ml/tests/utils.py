import inspect
from collections.abc import Iterable
from pathlib import Path

from nxontology.node import NodeInfo

from nxontology_ml.data import get_efo_otar_slim


def get_test_resource_path(p: Path | str) -> Path:
    if isinstance(p, str):
        # NOTE: The `inspect.stack` call cannot be nested
        calling_file_path = Path((inspect.stack()[1])[1])
        p = calling_file_path.parent / "test_resources" / p
    return p


def read_test_resource(p: Path | str) -> str:
    if isinstance(p, str):
        # NOTE: The `inspect.stack` call cannot be nested
        calling_file_path = Path((inspect.stack()[1])[1])
        p = calling_file_path.parent / "test_resources" / p
    test_resource = get_test_resource_path(p)
    assert test_resource.is_file(), f"Test resources: {test_resource} does not exist."
    return test_resource.read_text()


def get_test_nodes() -> Iterable[NodeInfo[str]]:
    test_efo_otar = get_test_resource_path("sampled_efo_otar_slim.json")
    nxo = get_efo_otar_slim(url=test_efo_otar.as_uri())
    yield from (nxo.node_info(node) for node in sorted(nxo.graph))
