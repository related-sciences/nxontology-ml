import json
from collections import Counter
from typing import Any, ParamSpecKwargs

from nxontology_ml.gpt_tagger._cache import _Cache
from nxontology_ml.gpt_tagger._chat_completion_middleware import (
    _ChatCompletionMiddleware,
)
from nxontology_ml.gpt_tagger._gpt_tagger import GptTagger
from nxontology_ml.gpt_tagger._models import TaskConfig
from nxontology_ml.gpt_tagger._openai_models import Response
from nxontology_ml.gpt_tagger._tiktoken_batcher import _TiktokenBatcher
from nxontology_ml.tests.utils import read_test_resource


def sanitize_json_format(s: str | dict[str, Any]) -> str:
    """Remove formatting & sort keys"""
    if isinstance(s, str):
        s = json.loads(s)
    assert isinstance(s, dict)
    return json.dumps(dict(sorted(s.items())))


def mk_stub_ccm(
    config: TaskConfig,
    stub_content: dict[str, Response] | None = None,
    counter: Counter[str] | None = None,
) -> _ChatCompletionMiddleware:
    if not stub_content:
        stub_payload_json = read_test_resource("precision_payload.json")
        stub_resp = Response(**json.loads(read_test_resource("precision_resp.json")))  # type: ignore
        stub_content = {sanitize_json_format(stub_payload_json): stub_resp}
    ccm = _ChatCompletionMiddleware.from_config(config, counter)
    # Remove json formatting
    stub_content = {sanitize_json_format(k): v for k, v in stub_content.items()}

    def create_fn_stub(**kwargs: ParamSpecKwargs) -> Response:
        return stub_content[sanitize_json_format(kwargs)]

    ccm._create_fn = create_fn_stub  # type: ignore
    return ccm


def mk_test_gpt_tagger(
    config: TaskConfig,
    cache_content: dict[str, bytes],
    stub_content: dict[str, Response] | None = None,
) -> GptTagger:
    """
    Helper to build test GptTagger instances
    """
    counter: Counter[str] = Counter()
    return GptTagger(
        chat_completion_middleware=mk_stub_ccm(
            config=config,
            stub_content=stub_content,
            counter=counter,
        ),
        cache=_Cache(
            storage=cache_content,
            key_hash_fn=config.cache_key_hash_fn,
            counter=counter,
        ),
        batcher=_TiktokenBatcher.from_config(config),
        config=config,
        counter=counter,
    )
