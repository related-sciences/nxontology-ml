from pathlib import Path
from tempfile import TemporaryDirectory

from nxontology_ml.efo import EFO_OTAR_SLIM_URL, write_efo_features
from tests.utils import get_test_resource_path, read_test_resource


def test_write_efo_features() -> None:
    # FIXME: Overly high level test
    assert (
        EFO_OTAR_SLIM_URL
        == "https://github.com/related-sciences/nxontology-data/raw/f0e450fe3096c3b82bf531bc5125f0f7e916aad8/efo_otar_slim.json"
    )

    with TemporaryDirectory() as tmpdir:
        test_efo_otar = get_test_resource_path("sampled_efo_otar_slim.json")
        expected_file = Path(tmpdir) / "output/efo_otar_slim/efo_otar_slim_features.tsv"
        assert not expected_file.exists()
        write_efo_features(url=test_efo_otar.as_uri(), parent_dir=Path(tmpdir))
        assert expected_file.exists()
        assert expected_file.read_text() == read_test_resource(
            "sampled_efo_otar_slim_features.tsv"
        )
