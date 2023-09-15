import json
import logging
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable
from copy import deepcopy

from nxontology.node import NodeInfo, NodeT

from nxontology_ml.gpt_tagger._cache import _Cache
from nxontology_ml.gpt_tagger._chat_completion_middleware import (
    _ChatCompletionMiddleware,
)
from nxontology_ml.gpt_tagger._models import LabelledNode, TaskConfig
from nxontology_ml.gpt_tagger._tiktoken_batcher import _TiktokenBatcher
from nxontology_ml.gpt_tagger._utils import (
    counter_or_empty,
    efo_id_from_yaml,
    node_efo_id,
    node_to_str_fn,
    parse_model_output,
)


class GptTagger:
    """
    Main entry point for node tagging using GPT models.
    Fetches labels for given nodes.

    Implementation details:
        The tagger relies on the following classes:
            - :class:`_ChatCompletionMiddleware`: Handle communications with the Open AI API.
            - :class:`_Cache`: Allows caching of the labels that have already been fetched. By default, the caches are
                on disk in the `./.cache` directory.
            - :class:`_TiktokenBatcher`: This utility class allows to count the amount of token of the records to be
                labelled. This token count is used for record batching.
    """

    def __init__(
        self,
        chat_completion_middleware: _ChatCompletionMiddleware,
        cache: _Cache,
        batcher: _TiktokenBatcher,
        config: TaskConfig,
        counter: Counter[str],
    ):
        """
        Intended to get constructed using cls.from_config(config)
        """
        self._chat_completion_middleware = chat_completion_middleware
        self._cache = cache
        self._batcher = batcher
        self._config = config
        self._counter = counter

    def fetch_labels(self, nodes: Iterable[NodeInfo[NodeT]]) -> Iterable[LabelledNode]:
        """
        Fetch the labels (using GPT) for the provided nodes.
        """
        buffer: list[str] | None
        for node in nodes:
            node_str = node_to_str_fn(self._config)(node)
            labels_str = self._cache.get(node_str)
            if labels_str:
                labels: list[str] = json.loads(labels_str)
                # Cache hit
                self._counter["Cache/hits"] += 1
                node_id = node_efo_id(node)
                logging.debug(f"Retrieved {node_id} from cache.")
                yield LabelledNode(node_efo_id=node_id, labels=labels)
            else:
                # Cache miss
                self._counter["Cache/misses"] += 1
                buffer = self._batcher.add_record_to_buffer(node_str)
                if buffer and len(buffer) > 0:
                    yield from self._do_fetch_record_batch(buffer)
        buffer = self._batcher.flush_buffer()
        if len(buffer) > 0:
            yield from self._do_fetch_record_batch(buffer)

    def _do_fetch_record_batch(  # noqa: C901
        self, records: list[str]
    ) -> Iterable[LabelledNode]:
        """
        Actually request labels for nodes
        """
        resp = self._chat_completion_middleware.create(records)

        for k, v in resp["usage"].items():
            self._counter[f"ChatCompletion/{k}"] += v  # type: ignore

        for choice in resp["choices"]:
            finish_reason = choice.get("finish_reason", "")
            if finish_reason == "length":
                raise ValueError(
                    f"The max number of completion tokens available was reached & the response has been truncated. "
                    f"Hint: You might want to shorten your prompt and/or lower the config's `prompt_token_ratio` "
                    f"value. Response: \n{resp}"
                )

        # If the model's `n` is >1, we will get several completions ("choices")
        choice_cnt = len(resp["choices"])

        # Extract labels from each choice
        labels_by_nodes: dict[str, list[str]] = defaultdict(list)
        for choice in resp["choices"]:
            for node_id, label in parse_model_output(
                choice["message"]["content"].splitlines()
            ):
                label = label.lower()
                if (
                    self._config.allowed_labels
                    and label not in self._config.allowed_labels
                ):
                    warnings.warn(
                        f"Label '{label}' does not belong to `allowed_labels`.",
                        stacklevel=1,
                    )
                else:
                    labels_by_nodes[node_id].append(label)

        # Handle missing & valid records
        rerun_warn_msg = " Hint: It is strongly recommended to rerun node tagging to fix the inconsistent nodes."
        for record in records:
            record_id = efo_id_from_yaml(record)
            labels = labels_by_nodes.get(record_id, [])
            if len(labels) == 0:
                warnings.warn(
                    f"Node {record_id} missing from response." + rerun_warn_msg,
                    stacklevel=1,
                )
            elif len(labels) < choice_cnt:
                warnings.warn(
                    f"Node {record_id} missing from some choices." + rerun_warn_msg,
                    stacklevel=1,
                )
            elif len(labels) > choice_cnt:
                warnings.warn(
                    f"Node {record_id} was duplicated." + rerun_warn_msg,
                    stacklevel=1,
                )
            else:
                # Valid response for this node
                self._cache[record] = json.dumps(sorted(labels))
                yield LabelledNode(record_id, labels)

        # Handle extra records
        for record in set(labels_by_nodes.keys()) - {
            efo_id_from_yaml(r) for r in records
        }:
            warnings.warn(
                f"Node {record} was part of this output but shouldn't be."
                + rerun_warn_msg,
                stacklevel=1,
            )

    def get_metrics(self, defensive_copy: bool = True) -> Counter[str]:
        """
        Return the metrics of the Tagger (and its dependencies)
        """
        if defensive_copy:
            return deepcopy(self._counter)
        return self._counter

    @classmethod
    def from_config(
        cls, config: TaskConfig, counter: Counter[str] | None = None
    ) -> "GptTagger":
        """Builds the fetcher (and dependencies) from TaskConfig"""
        counter = counter_or_empty(counter)
        return cls(
            chat_completion_middleware=_ChatCompletionMiddleware.from_config(
                config, counter
            ),
            cache=_Cache.from_config(config, counter),
            batcher=_TiktokenBatcher.from_config(config),
            config=config,
            counter=counter,
        )
