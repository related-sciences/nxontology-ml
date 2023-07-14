import pandas as pd
from nxontology.examples import create_metal_nxo

from nxontology_ml.features import NxontologyFeatures
from tests.utils import read_test_resource


def test_NxontologyFeatures() -> None:
    # FIXME: Overly high level test
    test_nxo = create_metal_nxo()
    test_nxo_features: NxontologyFeatures[str] = NxontologyFeatures(test_nxo.graph)
    test_df = test_nxo_features.get_features_df()
    expected_df = pd.read_json(read_test_resource("metal_nxo_features.json"))
    pd.testing.assert_frame_equal(
        test_df, expected_df, check_dtype=False
    )  # Json serialization messes up dtypes :(
