from nxontology_ml.data import get_efo_otar_slim, read_training_data
from nxontology_ml.sklearn_transformer import NodeFeatures


def test_node_features() -> None:
    nxo = get_efo_otar_slim()
    X, _ = read_training_data(sort=True, take=10)
    nf = NodeFeatures.from_nodes([nxo.node_info(n) for n in X])
    assert len(nf) == 10
