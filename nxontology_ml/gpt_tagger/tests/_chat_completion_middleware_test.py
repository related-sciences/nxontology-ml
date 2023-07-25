import json
import re
from collections import Counter
from copy import copy
from unittest.mock import Mock

import pytest

from nxontology_ml.gpt_tagger._chat_completion_middleware import (
    _ChatCompletionMiddleware,
)
from nxontology_ml.gpt_tagger._openai_models import (
    ChatCompletionMessage,
    ChatCompletionsPayload,
    Response,
)
from nxontology_ml.gpt_tagger._utils import node_to_str_fn
from nxontology_ml.gpt_tagger.tests._utils import mk_stub_ccm, precision_config
from nxontology_ml.tests.utils import get_test_nodes, read_test_resource


def _mk_test_ccm(
    partial_payload: ChatCompletionsPayload | None = None,
    prompt_template: str | None = None,
) -> _ChatCompletionMiddleware:
    # Valid by default
    partial_payload = partial_payload or ChatCompletionsPayload(
        model="gpt-3.5-turbo",
        messages=[ChatCompletionMessage(role="foo", content="bar")],
    )
    return _ChatCompletionMiddleware(
        partial_payload=partial_payload,
        prompt_template=prompt_template or "foo {records} bar",
        create_fn=Mock,
        counter=Counter(),
    )


def test_ctor_verify() -> None:
    # Valid
    _mk_test_ccm()

    # Invalid tests:
    invalid_ccp = ChatCompletionsPayload(
        model="INVALID",
        messages=[],
    )

    with pytest.raises(ValueError, match="Unsupported OpenAI Model: INVALID"):
        _mk_test_ccm(partial_payload=invalid_ccp)

    invalid_ccp["model"] = "gpt-3.5-turbo"
    with pytest.raises(
        ValueError,
        match=re.escape("Invalid partial_payload: Should contain message(s)"),
    ):
        _mk_test_ccm(partial_payload=invalid_ccp)

    with pytest.raises(ValueError, match=re.escape("Invalid prompt provided")):
        _mk_test_ccm(prompt_template="foo")


def test_create() -> None:
    config = copy(precision_config)
    config.model_temperature = 1
    config.model_top_p = 2

    stub_payload_json = ChatCompletionsPayload(  # type: ignore
        **json.loads(read_test_resource("precision_payload.json"))
    )
    stub_payload_json["temperature"] = 1
    stub_payload_json["top_p"] = 2
    stub_resp = Response(**json.loads(read_test_resource("precision_resp.json")))  # type: ignore
    stub_content = {json.dumps(stub_payload_json): stub_resp}

    ccm = mk_stub_ccm(config=config, stub_content=stub_content)
    node_to_str = node_to_str_fn(config=config)
    resp = ccm.create(records=(node_to_str(n) for n in get_test_nodes()))
    stub_resp = Response(  # type: ignore
        **json.loads(read_test_resource("precision_resp.json")),
    )
    assert resp == stub_resp


def test_from_config() -> None:
    ccm = _ChatCompletionMiddleware.from_config(precision_config)
    assert ccm._partial_payload["model"] == "gpt-3.5-turbo"
    assert ccm._partial_payload["messages"][0]["content"] == "__PLACEHOLDER__"
