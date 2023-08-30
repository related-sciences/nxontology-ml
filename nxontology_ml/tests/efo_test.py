import pandas as pd
from nxontology import NXOntology
from sklearn.pipeline import make_pipeline

from nxontology_ml.data import get_efo_otar_slim
from nxontology_ml.efo import NodeXrefFeatures
from nxontology_ml.features import NodeInfoFeatures, PrepareNodeFeatures
from nxontology_ml.sklearn_transformer import NodeFeatures
from nxontology_ml.tests.utils import assert_frame_equal_to, get_test_resource_path


def test_node_xref_features() -> None:
    nxo: NXOntology[str] = get_efo_otar_slim(
        url=get_test_resource_path("sampled_efo_otar_slim.json").as_uri()
    )
    p = make_pipeline(PrepareNodeFeatures(nxo), NodeInfoFeatures(), NodeXrefFeatures())
    nf: NodeFeatures = p.transform(X=sorted(nxo.graph))
    df = pd.concat([nf.num_features, nf.cat_features], axis=1)
    assert_frame_equal_to(df, "sampled_efo_otar_slim_features.json")
