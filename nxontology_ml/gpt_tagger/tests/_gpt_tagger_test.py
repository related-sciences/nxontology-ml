import json
import re
from collections import Counter
from itertools import repeat
from unittest.mock import Mock
from warnings import WarningMessage

import pytest

from nxontology_ml.data import get_efo_otar_slim
from nxontology_ml.gpt_tagger._gpt_tagger import GptTagger
from nxontology_ml.gpt_tagger._models import LabelledNode, TaskConfig
from nxontology_ml.gpt_tagger._openai_models import Response
from nxontology_ml.gpt_tagger.tests._utils import mk_test_gpt_tagger
from nxontology_ml.tests.utils import get_test_nodes, read_test_resource


def test_fetch_labels(precision_config: TaskConfig) -> None:
    cache_content: dict[str, bytes] = {}
    tagger = mk_test_gpt_tagger(precision_config, cache_content)
    labels = tagger.fetch_labels(get_test_nodes())
    assert list(labels) == [
        LabelledNode(node_efo_id="DOID:0050890", labels=["medium"]),
        LabelledNode(node_efo_id="EFO:0006792", labels=["medium"]),
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
        "/7665404d4f2728a09ed26b8ebf2b3be612bd7da2": b'["medium"]',
        "/962b25d69f79f600f23a17e2c3fe79948013b4de": b'["medium"]',
    }


def test_fetch_labels_cached(precision_config: TaskConfig) -> None:
    # Pre-loaded cache
    cache_content = {
        "/7665404d4f2728a09ed26b8ebf2b3be612bd7da2": b'["medium"]',
        "/962b25d69f79f600f23a17e2c3fe79948013b4de": b'["medium"]',
    }
    tagger = mk_test_gpt_tagger(precision_config, cache_content)
    labels = tagger.fetch_labels(get_test_nodes())
    assert list(labels) == [
        LabelledNode(node_efo_id="DOID:0050890", labels=["medium"]),
        LabelledNode(node_efo_id="EFO:0006792", labels=["medium"]),
    ]
    assert tagger.get_metrics() == Counter({"Cache/get": 2, "Cache/hits": 2})


def test_fetch_many_records(precision_config: TaskConfig) -> None:
    # Disable caching
    class PassthroughDict(dict[str, bytes]):
        def __setitem__(self, key: str, value: bytes) -> None:
            return

    tagger = mk_test_gpt_tagger(precision_config, cache_content=PassthroughDict())

    def _r(n: int) -> Response:
        r = json.loads(read_test_resource("precision_resp.json"))
        r["choices"][0]["message"]["content"] = "\n".join(
            ["id|precision"] + ["DOID:0050890|medium"] * n
        )
        return Response(r)  # type: ignore

    # Need the responses to match the size of the requests
    tagger._chat_completion_middleware._create_fn = Mock(
        side_effect=[_r(33), _r(33), _r(33), _r(1)]
    )

    test_node = next(iter(get_test_nodes()))
    list(tagger.fetch_labels(repeat(test_node, 100)))
    assert tagger.get_metrics() == Counter(
        {
            "ChatCompletion/total_tokens": 4492,
            "ChatCompletion/prompt_tokens": 4408,
            "Cache/get": 100,
            "Cache/misses": 100,
            "ChatCompletion/records_processed": 100,
            "Cache/set": 1,
            "ChatCompletion/completion_tokens": 84,
            "ChatCompletion/create_requests": 4,
        }
    )


def test_get_metrics(precision_config: TaskConfig) -> None:
    tagger = mk_test_gpt_tagger(precision_config, cache_content={})
    tagger._counter["test"] += 42

    # Defensive copy: No effect
    counter = tagger.get_metrics()
    counter["test"] += 1
    assert tagger.get_metrics() == Counter({"test": 42})

    # No defensive copy
    counter = tagger.get_metrics(defensive_copy=False)
    counter["test"] += 1
    assert tagger.get_metrics() == Counter({"test": 43})


def test_from_config(precision_config: TaskConfig) -> None:
    counter: Counter[str] = Counter()
    tagger = GptTagger.from_config(precision_config, counter=counter)

    # Check that the counter is actually passed to all dependencies
    counter_id = id(counter)
    assert id(tagger._counter) == counter_id
    assert id(tagger._chat_completion_middleware._counter) == counter_id
    assert id(tagger._cache._counter) == counter_id


def test_resp_truncated(precision_config: TaskConfig) -> None:
    stub_resp = Response(**json.loads(read_test_resource("precision_resp.json")))  # type: ignore[misc]
    assert stub_resp["choices"][0]["finish_reason"] == "stop"
    stub_resp["choices"][0]["finish_reason"] = "length"  # Simulate resp truncation
    expected_req = read_test_resource("precision_payload.json")
    tagger = mk_test_gpt_tagger(
        config=precision_config,
        stub_content={expected_req: stub_resp},
        cache_content={},
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The max number of completion tokens available was reached & the response has been truncated. Hint: "
        ),
    ):
        list(tagger.fetch_labels(get_test_nodes()))


def _assert_user_warning_starts_with(warn: WarningMessage, s: str) -> None:
    assert isinstance(warn.message, UserWarning)
    warn_msg = warn.message.args[0]
    assert isinstance(warn_msg, str)
    assert warn_msg.startswith(s)


def test_resp_id_mismatch(precision_config: TaskConfig) -> None:
    expected_req = read_test_resource("mismatch_payload.json")
    stub_resp = Response(**json.loads(read_test_resource("mismatch_resp.json")))  # type: ignore[misc]
    tagger = mk_test_gpt_tagger(
        config=precision_config,
        stub_content={expected_req: stub_resp},
        cache_content={},
    )
    nxo = get_efo_otar_slim()
    valid_resp_node = "DOID:0050890"
    missing_one_node = "EFO:0006793"
    missing_all_node = "EFO:0006794"
    duplicated_node = "EFO:0006795"
    wrong_label_node = "EFO:0006796"
    nodes = [
        nxo.node_info(n)
        for n in [
            valid_resp_node,
            missing_one_node,
            missing_all_node,
            duplicated_node,
            wrong_label_node,
        ]
    ]
    with pytest.warns() as warns:
        output = list(tagger.fetch_labels(nodes))
        expected_output = [
            LabelledNode(node_efo_id="DOID:0050890", labels=["medium", "high"])
        ]
        assert output == expected_output

        # Verify warnings
        assert len(warns) == 7
        warns = sorted(warns, key=lambda w: w.message.args[0])  # type: ignore
        _assert_user_warning_starts_with(
            warns[0], "Label 'na' does not belong to `allowed_labels`."
        )
        _assert_user_warning_starts_with(
            warns[1], "Node EFO:0000206 was part of this output but shouldn't be."
        )
        _assert_user_warning_starts_with(
            warns[2], "Node EFO:0006792 was part of this output but shouldn't be."
        )
        _assert_user_warning_starts_with(
            warns[3], "Node EFO:0006793 missing from some choices"
        )
        _assert_user_warning_starts_with(
            warns[4], "Node EFO:0006794 missing from response"
        )
        _assert_user_warning_starts_with(warns[5], "Node EFO:0006795 was duplicated")
        _assert_user_warning_starts_with(
            warns[6], "Node EFO:0006796 missing from some choices"
        )
