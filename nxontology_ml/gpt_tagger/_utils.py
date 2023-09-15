import sys
from collections import Counter
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any
from urllib.parse import quote_plus

import yaml
from nxontology.node import NodeInfo, NodeT
from yaml import Loader

from nxontology_ml.gpt_tagger._models import TaskConfig


def config_to_cache_namespace(config: TaskConfig) -> str:
    """
    Compute cache namespace based on task name and  prompt_version.

    NOTE:
        - Neither the prompt path nor content are part of the cache namespace
        - Rationale: Allow the prompt to be changeable / movable independent of content & location
    """
    return quote_plus(f"{config.name}_{config.prompt_version}_n{config.model_n}")


def node_to_str_fn(
    config: TaskConfig, rm_efo_prefix: bool = True
) -> Callable[[NodeInfo[NodeT]], str]:
    """
    Returns a function to convert an EFO node into a string (serialized YAML dict of features)
    """

    def f(node: NodeInfo[NodeT]) -> str:
        node_data: dict[str, Any] = node.data
        filtered = ((attr, node_data[attr]) for attr in config.node_attributes)
        if rm_efo_prefix:
            filtered = ((k.removeprefix("efo_"), v) for k, v in filtered)
        s = yaml.dump([dict(filtered)], width=sys.maxsize, sort_keys=False).strip()
        assert isinstance(s, str)
        return s

    return f


def parse_model_output(
    model_output: Iterable[str], skip_header: bool = True
) -> Iterable[tuple[str, str]]:
    # At the moment, only support "val1|val2"
    start_idx = 1 if skip_header else 0
    for line in islice(model_output, start_idx, None):
        if len(line.strip()) == 0:
            continue
        tokens = line.split(sep="|", maxsplit=2)
        assert len(tokens) == 2, f"Not enough values to unpack from '{line}'"
        node_id, label = tokens
        yield node_id, label


def node_efo_id(node: NodeInfo[NodeT]) -> str:
    # TODO: Check efo_id coverage in data
    node_id = node.data["efo_id"]
    assert isinstance(node_id, str)
    return node_id


def efo_id_from_yaml(yaml_record: str, rm_efo_prefix: bool = True) -> str:
    parsed_rec: list[dict[str, str]] = yaml.load(yaml_record, Loader)
    assert (
        len(parsed_rec) == 1
    ), f"Only one record should be parsed, got: {len(parsed_rec)}"
    key = "id" if rm_efo_prefix else "efo_id"
    return parsed_rec[0][key]


def counter_or_empty(counter: Counter[str] | None) -> Counter[str]:
    """
    Note: Cannot do "counter = counter or Counter()" because Counter overrides __or__
    """
    if counter is None:
        return Counter()
    return counter
