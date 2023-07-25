from pathlib import Path
from tempfile import TemporaryDirectory

from nxontology_ml.gpt_tagger._cache import _Cache, _LazyLSM
from nxontology_ml.gpt_tagger.tests._utils import precision_config
from nxontology_ml.utils import ROOT_DIR


def test_from_config() -> None:
    expected_cache_path = ROOT_DIR / ".cache/precision_v1.ldb"
    cache = _Cache.from_config(precision_config)
    assert isinstance(cache._storage, _LazyLSM)
    assert Path(cache._storage._filename) == expected_cache_path
    assert cache._key_hash_fn == "sha1"
    assert cache._namespace == ""


def test_main() -> None:
    with TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "precision_v1.ldb"
        cache = _Cache.from_config(precision_config, cache_path=cache_path)

        assert cache.get("KEY", "DEFAULT") == "DEFAULT"
        cache["KEY"] = "value"
        assert cache.get("KEY", "DEFAULT") == "value"

        cache2 = _Cache.from_config(precision_config, cache_path=cache_path)
        cache2["KEY"] = "value"
        del cache2["KEY"]
        assert cache2.get("KEY", "DEFAULT") == "DEFAULT"


def test_LazyLSM() -> None:
    with TemporaryDirectory() as tmpdir:
        with _LazyLSM(filename=tmpdir + "/test.ldb") as llsm:
            assert len(llsm) == 0
            llsm["foo"] = "bar"
            assert len(llsm) == 1
            del llsm["foo"]
            assert len(llsm) == 0
