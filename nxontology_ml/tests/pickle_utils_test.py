import pickle

from nxontology_ml.data import EFO_OTAR_SLIM_URL
from nxontology_ml.pickle_utils import HostedNXOntology


def test_nxo_graph_roundtrip() -> None:
    nxo = HostedNXOntology.from_url(url=EFO_OTAR_SLIM_URL)
    # To describe the pickled object to stdout, run:
    # `pickletools.dis(pickle.dumps(nxo), annotate=1)`
    nxo_roundtrip = pickle.loads(pickle.dumps(nxo))
    assert set(nxo.graph) == set(nxo_roundtrip.graph)
