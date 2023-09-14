from textwrap import dedent

from nxontology_ml.gpt_tagger._utils import (
    config_to_cache_namespace,
    efo_id_from_yaml,
    node_efo_id,
    node_to_str_fn,
    parse_model_output,
)
from nxontology_ml.gpt_tagger.tests._utils import precision_config
from nxontology_ml.tests.utils import get_test_nodes


def test_config_to_cache_namespace() -> None:
    assert config_to_cache_namespace(precision_config) == "precision_v1_n1"


def test_node_to_str_fn() -> None:
    test_nodes = get_test_nodes()
    fn = node_to_str_fn(precision_config)
    expected_str = """\
- id: DOID:0050890
  label: synucleinopathy
  definition: A neurodegenerative disease that is characterized by the abnormal accumulation of aggregates of alpha-synuclein protein in neurons, nerve fibres or glial cells. [url:http://en.wikipedia.org/wiki/Synucleinopathies ]"""
    assert fn(next(iter(test_nodes))) == expected_str


def test_parse_model_output() -> None:
    example_model_output = dedent(
        """\
    id|precision
    MONDO:0014498|high
    EFO:0009011|medium
    MONDO:0024239|low
    """
    )
    parsed_out = list(parse_model_output(example_model_output.splitlines()))
    assert parsed_out == [
        ("MONDO:0014498", "high"),
        ("EFO:0009011", "medium"),
        ("MONDO:0024239", "low"),
    ]


def test_node_efo_id() -> None:
    node = next(iter(get_test_nodes()))
    assert node_efo_id(node) == "DOID:0050890"


def test_efo_id_from_yaml() -> None:
    node = next(iter(get_test_nodes()))
    fn = node_to_str_fn(precision_config)
    assert efo_id_from_yaml(fn(node)) == "DOID:0050890"
