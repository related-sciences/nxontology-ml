from pathlib import Path
from tempfile import TemporaryDirectory

from nxontology_ml.gpt_tagger import TaskConfig
from nxontology_ml.gpt_tagger._cache import LazyLSM, _Cache
from nxontology_ml.utils import ROOT_DIR


def test_from_config() -> None:
    config = TaskConfig(
        name="precision",
        prompt_path=ROOT_DIR / "prompts/precision_v1.txt",
        openai_model_name="gpt-4",
        node_attributes=["efo_id", "efo_label", "efo_definition"],
        model_n=3,
    )
    expected_cache_path = Path("/tmp/nxontology-ml/cache/precision_v1_n3.ldb")
    cache = _Cache.from_config(config)
    assert isinstance(cache._storage, LazyLSM)
    assert Path(cache._storage._filename) == expected_cache_path
    assert cache._key_hash_fn == "sha1"
    assert cache._namespace == ""


def test_main(precision_config: TaskConfig) -> None:
    cache = _Cache.from_config(precision_config)

    assert cache.get("KEY", "DEFAULT") == "DEFAULT"
    cache["KEY"] = "value"
    assert cache.get("KEY", "DEFAULT") == "value"

    cache2 = _Cache.from_config(precision_config)
    cache2["KEY"] = "value"
    del cache2["KEY"]
    assert cache2.get("KEY", "DEFAULT") == "DEFAULT"


def test_LazyLSM() -> None:
    with TemporaryDirectory() as tmpdir:
        with LazyLSM(filename=tmpdir + "/test.ldb") as llsm:
            assert len(llsm) == 0
            llsm["foo"] = "bar"
            assert len(llsm) == 1
            del llsm["foo"]
            assert len(llsm) == 0
