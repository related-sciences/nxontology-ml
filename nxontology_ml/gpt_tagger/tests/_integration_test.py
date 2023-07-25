import json
import logging
from pprint import pprint

import pytest

from nxontology_ml.efo import get_efo_otar_slim
from nxontology_ml.gpt_tagger._chat_completion_middleware import (
    _ChatCompletionMiddleware,
)
from nxontology_ml.gpt_tagger._gpt_tagger import GptTagger
from nxontology_ml.gpt_tagger._models import TaskConfig
from nxontology_ml.gpt_tagger._openai_models import Response
from nxontology_ml.gpt_tagger._utils import node_to_str_fn
from nxontology_ml.gpt_tagger.tests._utils import precision_config
from nxontology_ml.tests.utils import get_test_nodes, read_test_resource
from nxontology_ml.utils import ROOT_DIR


@pytest.mark.skip(reason="IT: Makes a real openai api call")
def test_chat_completion_precision_it() -> None:
    # NOTE: Flaky API response, even with temp=0 :(
    # NOTE: Needs an OPENAI_API_KEY setup, see main README.md
    ccm = _ChatCompletionMiddleware.from_config(precision_config)
    node_to_str = node_to_str_fn(config=precision_config)
    resp = ccm.create(records=(node_to_str(n) for n in get_test_nodes()))
    expected_resp = Response(**json.loads(read_test_resource("precision_resp.json")))  # type: ignore
    assert resp["model"] == "gpt-3.5-turbo-0613"
    assert resp["choices"] == expected_resp["choices"]
    assert resp["usage"] == expected_resp["usage"]


@pytest.mark.skip(reason="IT: Makes a real openai api call")
def test_readme_code_it() -> None:
    # Simply test that it shouldn't crash? :(
    logging.basicConfig(level=logging.DEBUG)
    # Create a config for EFO nodes labelling
    config = TaskConfig(
        name="precision",
        prompt_path=ROOT_DIR / "prompts/precision_v1.txt",
        openai_model_name="gpt-4",
        node_attributes=["efo_id", "efo_label", "efo_definition"],
    )

    # Get a few EFO nodes
    nxo = get_efo_otar_slim()
    nodes = (nxo.node_info(node) for node in sorted(nxo.graph)[:20])

    # Get their labels
    tagger = GptTagger.from_config(config)
    for ln in tagger.fetch_labels(nodes):
        print(f"{ln.node_efo_id}: {ln.label}")

    # Inspect metrics
    print("\nTagger metrics:")
    pprint(tagger.get_metrics())
