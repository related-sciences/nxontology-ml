import json

from nxontology.examples import create_metal_nxo
from sklearn.pipeline import make_pipeline

from nxontology_ml.features import NodeInfoFeatures, PrepareNodeFeatures
from nxontology_ml.sklearn_transformer import NodeFeatures
from nxontology_ml.tests.utils import read_test_resource


def test_node_info_features() -> None:
    nxo = create_metal_nxo()
    p = make_pipeline(PrepareNodeFeatures(nxo), NodeInfoFeatures())
    nf: NodeFeatures = p.transform(X=sorted(nxo.graph))
    features = nf.num_features.to_dict(orient="records")
    expected_features = json.loads(read_test_resource("metal_nxo_features.json"))
    assert features == expected_features
