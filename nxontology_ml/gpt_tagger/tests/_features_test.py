import json

import pandas as pd
import pytest
from nxontology import NXOntology
from pandas._testing import assert_frame_equal
from sklearn.pipeline import make_pipeline

from nxontology_ml.data import get_efo_otar_slim
from nxontology_ml.features import PrepareNodeFeatures
from nxontology_ml.gpt_tagger import GptTagger
from nxontology_ml.gpt_tagger._features import DEFAULT_CONF, GptTagFeatures
from nxontology_ml.gpt_tagger._openai_models import Response
from nxontology_ml.gpt_tagger.tests._utils import mk_test_gpt_tagger, precision_config
from nxontology_ml.sklearn_transformer import NodeFeatures
from nxontology_ml.tests.utils import read_test_resource
from nxontology_ml.utils import ROOT_DIR


@pytest.fixture
def sampled_nxo() -> NXOntology[str]:
    return get_efo_otar_slim(
        url=ROOT_DIR.joinpath(
            "nxontology_ml/tests/test_resources/sampled_efo_otar_slim.json"
        ).as_uri()
    )


@pytest.fixture
def tagger() -> GptTagger:
    expected_req = read_test_resource("precision_payload.json")
    stub_resp = Response(**json.loads(read_test_resource("precision_resp.json")))  # type: ignore[misc]
    return mk_test_gpt_tagger(stub_content={expected_req: stub_resp}, cache_content={})


def test_transform(tagger: GptTagger, sampled_nxo: NXOntology[str]) -> None:
    p = make_pipeline(
        PrepareNodeFeatures(sampled_nxo),
        GptTagFeatures(
            enabled=True,
            tagger=tagger,
            config=precision_config,
        ),
    )
    nf: NodeFeatures = p.transform(X=sorted(sampled_nxo.graph))
    df = pd.concat([nf.num_features, nf.cat_features], axis=1)
    expected_df = pd.DataFrame(
        [{"gpt-3.5-turbo_tag_0": "medium"}, {"gpt-3.5-turbo_tag_0": "medium"}]
    )
    assert_frame_equal(df, expected_df)


def test_disabled(tagger: GptTagger, sampled_nxo: NXOntology[str]) -> None:
    p = make_pipeline(
        PrepareNodeFeatures(sampled_nxo),
        GptTagFeatures(
            enabled=False,
            tagger=tagger,
            config=precision_config,
        ),
    )
    nf: NodeFeatures = p.transform(X=sorted(sampled_nxo.graph))
    df = pd.concat([nf.num_features, nf.cat_features], axis=1)
    assert_frame_equal(df, pd.DataFrame([]))


def test_from_config() -> None:
    t = GptTagFeatures.from_config()
    assert t._config == DEFAULT_CONF
