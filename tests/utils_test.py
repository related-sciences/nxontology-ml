from pathlib import Path
from tempfile import TemporaryDirectory

from nxontology.examples import create_metal_nxo

from nxontology_ml.utils import ROOT_DIR, get_output_directory


def test_ROOT_DIR() -> None:
    # Presence of 'pyproject.toml' at the root should be reasonably stable
    assert ROOT_DIR.joinpath("pyproject.toml").is_file()


def test_get_output_directory() -> None:
    test_nxo = create_metal_nxo()
    assert test_nxo.name, "Test resources has changed, please update test."
    with TemporaryDirectory() as tmpdir:
        expected_dir = Path(tmpdir) / "output" / test_nxo.name
        assert not expected_dir.exists()
        out_dir = get_output_directory(
            nxo=test_nxo,
            parent_dir=Path(tmpdir),
        )
        assert out_dir == expected_dir
        assert expected_dir.exists()
