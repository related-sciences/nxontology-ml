from itertools import islice

from nxontology.examples import create_metal_nxo
from sklearn.pipeline import make_pipeline

from nxontology_ml.data import get_efo_otar_slim
from nxontology_ml.features import (
    NodeInfoFeatures,
    PrepareNodeFeatures,
    SubsetsFeatures,
    TherapeuticAreaFeatures,
)
from nxontology_ml.sklearn_transformer import NodeFeatures
from nxontology_ml.tests.utils import assert_frame_equal_to


def test_node_info_features() -> None:
    nxo = create_metal_nxo()
    p = make_pipeline(PrepareNodeFeatures(nxo), NodeInfoFeatures())
    nf: NodeFeatures = p.transform(X=sorted(nxo.graph))
    assert_frame_equal_to(nf.num_features, "metal_nxo_features.json")


def test_subset_features() -> None:
    nxo = get_efo_otar_slim()
    p = make_pipeline(PrepareNodeFeatures(nxo), SubsetsFeatures())
    nodes = [
        "MONDO:0009451",  # Has 0 subset
        "MONDO:0009452",  # Has 1 subset
        "MONDO:0016966",  # Has 2 subsets
    ]
    nf: NodeFeatures = p.transform(nodes)
    assert_frame_equal_to(nf.num_features, "subsets_features.json")


def test_ta_features() -> None:
    nxo = get_efo_otar_slim()
    p = make_pipeline(PrepareNodeFeatures(nxo), TherapeuticAreaFeatures())
    nf: NodeFeatures = p.transform(islice(sorted(nxo.graph), 5))
    assert nf.num_features.shape == (5, 20)
    assert nf.num_features.sparse.density == 0.08
    feature_dict = [
        {k: v for k, v in rec.items() if v != 0}
        for rec in nf.num_features.to_dict(orient="records")
    ]
    assert feature_dict == [
        {"TA_nervous_system_disease": 1, "TA_nutritional_or_metabolic_disease": 1},
        {"TA_infectious_disease": 1},
        {"TA_infectious_disease": 1, "TA_gastrointestinal_disease": 1},
        {
            "TA_musculoskeletal_or_connective_tissue_disease": 1,
            "TA_respiratory_or_thoracic_disease": 1,
        },
        {"TA_infectious_disease": 1},
    ]
