from typing import Any

import fsspec
import networkx as nx
from nxontology import NXOntology

from nxontology_ml.data import EFO_OTAR_SLIM_URL, get_efo_otar_slim


class HostedNXOntology(NXOntology[str]):
    """
    Ontology backed by a remotely hosted Json file (assumed immutable)
    (Main purpose: Ease serialization of NXOntology)
    """

    def __init__(
        self,
        url: str,
        graph: nx.DiGraph | None = None,
    ):
        super().__init__(graph=graph)
        self._url = url

    @classmethod
    def from_url(cls, url: str = EFO_OTAR_SLIM_URL) -> "HostedNXOntology":
        protocol = fsspec.get_fs_token_paths(url)[0].protocol
        assert protocol.startswith("http"), f"Unsupported protocol: {protocol}"
        return cls(url=url, graph=get_efo_otar_slim(url).graph)

    ##
    # Pickle logic
    _CURRENT_PICKLE_VERSION = "v1"

    def __getstate__(self) -> dict[str, Any]:
        return {"version": self._CURRENT_PICKLE_VERSION, "_url": self._url}

    def __setstate__(self, state: dict[str, Any]) -> None:
        assert state.get("version", None) == self._CURRENT_PICKLE_VERSION
        self.__dict__.update(self.from_url(state["_url"]).__dict__)
