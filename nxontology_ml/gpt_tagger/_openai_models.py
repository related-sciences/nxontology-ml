from dataclasses import dataclass
from typing import TypedDict

from typing_extensions import NotRequired


class ChatCompletionMessage(TypedDict):
    role: str
    content: str


class ChatCompletionsPayload(TypedDict):
    """
    Typed payload for https://api.openai.com/v1/chat/completions

    Notes:
    - Mirrors https://platform.openai.com/docs/api-reference/chat/create
    - Only the field actually used are defined
    """

    model: str
    messages: list[ChatCompletionMessage]
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    n: NotRequired[int]


class Choice(TypedDict):
    index: int
    message: ChatCompletionMessage
    finish_reason: str


class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Response(TypedDict):
    """
    See https://platform.openai.com/docs/api-reference/chat/create
    """

    id: str
    model: str
    object: str
    create: int
    choices: list[Choice]
    usage: Usage


_4K = 4096
_8K = 8192
_16K = 16384
_32K = 32768


@dataclass
class _ModelConfig:
    """
    Config for Open AI models
    """

    # Must be an actual model
    name: str

    # Used for batching example, without exceeding prompt limits
    max_token_cnt: int


_OPENAI_MODELS: list[_ModelConfig] = [
    # https://platform.openai.com/docs/models/gpt-4
    _ModelConfig(name="gpt-4", max_token_cnt=_8K),
    _ModelConfig(name="gpt-4-0613", max_token_cnt=_8K),
    _ModelConfig(name="gpt-4-32k", max_token_cnt=_32K),
    _ModelConfig(name="gpt-4-32k-0613", max_token_cnt=_32K),
    # https://platform.openai.com/docs/models/gpt-3-5
    _ModelConfig(name="gpt-3.5-turbo", max_token_cnt=_4K),
    _ModelConfig(name="gpt-3.5-turbo-16k", max_token_cnt=_16K),
    _ModelConfig(name="gpt-3.5-turbo-0613", max_token_cnt=_4K),
    _ModelConfig(name="gpt-3.5-turbo-16k-0613", max_token_cnt=_16K),
    # Older models
    # https://platform.openai.com/docs/deprecations/
]
OPENAI_MODELS: dict[str, _ModelConfig] = {m.name: m for m in _OPENAI_MODELS}

OPENAI_API_KEY = "OPENAI_API_KEY"
