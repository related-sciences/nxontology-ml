from dataclasses import dataclass
from pathlib import Path


@dataclass
class TaskConfig:
    """
    Configuration for an EFO node labelling task using few-shots ChatGPT.

    Assumptions about the prompt content & expected model output:
    - Model:
        - GPT4 and 3.5 models are supported
    - Prompt:
        - Has a {records} field where the records get injected
        - The records will be injected as a YAML list of dictionaries
        - "efo_id" must be part of the record
    - Model output:
        - The first line of the output of the model will contain CSV headers
        - The output of the model will be "|" delimited CSV values
    """

    # Name of the task, should be unique (in tandem with prompt_version)
    name: str

    # Path to (templated) prompt to be used
    prompt_path: Path

    # (Ordered) attributes of the nodes that will be used to extract each node's "record"
    # "efo_id" must be present
    node_attributes: list[str]

    # Name of the OpenAI model (Note: A model config must be declared for this model)
    openai_model_name: str

    # Prompt version, used for cache invalidation
    # Note: The content of the cache could have been invalidated using some magic (e.g. hash of
    #   prompt content) but it's likely better to use manual cache invalidation
    prompt_version: str = "v1"

    # See https://platform.openai.com/docs/api-reference/chat/create#temperature
    model_temperature: float | None = None

    # See https://platform.openai.com/docs/api-reference/chat/create#top_p
    model_top_p: float | None = None

    # See https://platform.openai.com/docs/api-reference/chat/create#n
    model_n: int = 1

    # The max number of tokens in the prompt conditions how many token are left for the completion
    #   e.g. If using at most 6553 tokens (8192 * .8) for the prompt, 1639 tokens (8192 - 6553) are
    #   left for the completion
    prompt_token_ratio: float = (
        0.8  # By default, we use 80% of the max token count for the prompt
    )

    # Optionally hash cache key (string values for nodes)
    # The node features will be used as cache key if not hashed.
    cache_key_hash_fn: str | None = "sha1"

    # Allowed labels: If provided, all output labels must belong to this set
    allowed_labels: frozenset[str] | None = None


@dataclass
class LabelledNode:
    """
    Instance of labelled record (e.g. Output of model)
    """

    node_efo_id: str
    # If the model's `n` parameter is >1, then several completions are generated for each prompt
    labels: list[str]
