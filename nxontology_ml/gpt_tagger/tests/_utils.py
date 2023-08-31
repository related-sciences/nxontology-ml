import json
from collections import Counter
from typing import ParamSpecKwargs

from nxontology_ml.gpt_tagger._cache import _Cache
from nxontology_ml.gpt_tagger._chat_completion_middleware import (
    _ChatCompletionMiddleware,
)
from nxontology_ml.gpt_tagger._gpt_tagger import GptTagger
from nxontology_ml.gpt_tagger._models import TaskConfig
from nxontology_ml.gpt_tagger._openai_models import Response
from nxontology_ml.gpt_tagger._tiktoken_batcher import _TiktokenBatcher
from nxontology_ml.gpt_tagger._utils import node_to_str_fn
from nxontology_ml.tests.utils import get_test_resource_path, read_test_resource

precision_config = TaskConfig(
    name="precision",
    prompt_path=get_test_resource_path("precision_v1.txt"),
    node_attributes=["efo_id", "efo_label", "efo_definition"],
    openai_model_name="gpt-3.5-turbo",
    model_temperature=0,
)


def mk_stub_ccm(
    config: TaskConfig | None = None,
    stub_content: dict[str, Response] | None = None,
    counter: Counter[str] | None = None,
) -> _ChatCompletionMiddleware:
    if not config:
        config = precision_config
    if not stub_content:
        stub_payload_json = read_test_resource("precision_payload.json")
        stub_resp = Response(**json.loads(read_test_resource("precision_resp.json")))  # type: ignore
        stub_content = {stub_payload_json: stub_resp}
    ccm = _ChatCompletionMiddleware.from_config(config, counter)
    # Remove json formatting
    stub_content = {json.dumps(json.loads(k)): v for k, v in stub_content.items()}

    def create_fn_stub(**kwargs: ParamSpecKwargs) -> Response:
        return stub_content[json.dumps(kwargs)]

    ccm._create_fn = create_fn_stub  # type: ignore
    return ccm


def mk_test_gpt_tagger(
    cache_content: dict[str, bytes],
    stub_content: dict[str, Response] | None = None,
) -> GptTagger:
    """
    Helper to build test GptTagger instances
    """
    counter: Counter[str] = Counter()
    return GptTagger(
        chat_completion_middleware=mk_stub_ccm(
            counter=counter, stub_content=stub_content
        ),
        cache=_Cache(
            storage=cache_content,
            key_hash_fn=precision_config.cache_key_hash_fn,
            counter=counter,
        ),
        batcher=_TiktokenBatcher.from_config(precision_config),
        node_to_str_fn=node_to_str_fn(precision_config),
        counter=counter,
    )
