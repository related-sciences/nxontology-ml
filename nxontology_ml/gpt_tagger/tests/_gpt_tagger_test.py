import json
from collections import Counter
from itertools import repeat
from unittest.mock import Mock

import pytest

from nxontology_ml.gpt_tagger._gpt_tagger import GptTagger
from nxontology_ml.gpt_tagger._models import LabelledNode
from nxontology_ml.gpt_tagger._openai_models import Response
from nxontology_ml.gpt_tagger.tests._utils import mk_test_gpt_tagger, precision_config
from nxontology_ml.tests.utils import get_test_nodes, read_test_resource


def test_fetch_labels() -> None:
    cache_content: dict[str, str] = {}
    tagger = mk_test_gpt_tagger(cache_content)
    labels = tagger.fetch_labels(get_test_nodes())
    assert list(labels) == [
        LabelledNode(node_efo_id="DOID:0050890", label="medium"),
        LabelledNode(node_efo_id="EFO:0006792", label="medium"),
    ]
    assert tagger.get_metrics() == Counter(
        {
            "Cache/get": 2,
            "Cache/misses": 2,
            "Cache/set": 2,
            "ChatCompletion/completion_tokens": 21,
            "ChatCompletion/create_requests": 1,
            "ChatCompletion/prompt_tokens": 1102,
            "ChatCompletion/records_processed": 2,
            "ChatCompletion/total_tokens": 1123,
        }
    )
    assert cache_content == {
        "/7665404d4f2728a09ed26b8ebf2b3be612bd7da2": "medium",
        "/962b25d69f79f600f23a17e2c3fe79948013b4de": "medium",
    }


def test_fetch_labels_cached() -> None:
    # Pre-loaded cache
    cache_content = {
        "/7665404d4f2728a09ed26b8ebf2b3be612bd7da2": "medium",
        "/962b25d69f79f600f23a17e2c3fe79948013b4de": "medium",
    }
    tagger = mk_test_gpt_tagger(cache_content)
    labels = tagger.fetch_labels(get_test_nodes())
    assert list(labels) == [
        LabelledNode(node_efo_id="DOID:0050890", label="medium"),
        LabelledNode(node_efo_id="EFO:0006792", label="medium"),
    ]
    assert tagger.get_metrics() == Counter({"Cache/get": 2, "Cache/hits": 2})


def test_fetch_many_records() -> None:
    # Disable caching
    class PassthroughDict(dict[str, str]):
        def __setitem__(self, key: str, value: str) -> None:
            return

    tagger = mk_test_gpt_tagger(cache_content=PassthroughDict())

    def _r(n: int) -> Response:
        r = json.loads(read_test_resource("precision_resp.json"))
        r["choices"][0]["message"]["content"] = "\n".join(
            ["id|precision"] + ["DOID:0050890|medium"] * n
        )
        return Response(r)  # type: ignore

    # Need the responses to match the size of the requests
    tagger._chat_completion_middleware._create_fn = Mock(
        side_effect=[_r(45), _r(45), _r(10)]
    )

    test_node = next(iter(get_test_nodes()))
    list(tagger.fetch_labels(repeat(test_node, 100)))
    assert tagger.get_metrics() == Counter(
        {
            "ChatCompletion/total_tokens": 3369,
            "ChatCompletion/prompt_tokens": 3306,
            "Cache/get": 100,
            "Cache/misses": 100,
            "ChatCompletion/records_processed": 100,
            "Cache/set": 100,
            "ChatCompletion/completion_tokens": 63,
            "ChatCompletion/create_requests": 3,
        }
    )


def test_get_metrics() -> None:
    tagger = mk_test_gpt_tagger(cache_content={})
    tagger._counter["test"] += 42

    # Defensive copy: No effect
    counter = tagger.get_metrics()
    counter["test"] += 1
    assert tagger.get_metrics() == Counter({"test": 42})

    # No defensive copy
    counter = tagger.get_metrics(defensive_copy=False)
    counter["test"] += 1
    assert tagger.get_metrics() == Counter({"test": 43})


def test_malformed_response() -> None:
    stub_payload_json = read_test_resource("precision_payload.json")
    stub_resp = Response(**json.loads(read_test_resource("precision_resp.json")))  # type: ignore
    stub_resp["choices"] = []
    stub_content = {stub_payload_json: stub_resp}
    tagger = mk_test_gpt_tagger(cache_content={}, stub_content=stub_content)
    with pytest.raises(ValueError, match="The response should have only one 'choice'"):
        list(tagger.fetch_labels(get_test_nodes()))


def test_from_config() -> None:
    counter: Counter[str] = Counter()
    tagger = GptTagger.from_config(precision_config, counter=counter)

    # Check that the counter is actually passed to all dependencies
    counter_id = id(counter)
    assert id(tagger._counter) == counter_id
    assert id(tagger._chat_completion_middleware._counter) == counter_id
    assert id(tagger._cache._counter) == counter_id
