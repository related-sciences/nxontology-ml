from pathlib import Path

TEST_RESOURCES_DIR = Path(__file__).parent / "test_resources"


def get_test_resource_path(p: Path | str) -> Path:
    if isinstance(p, str):
        p = Path(p)
    return TEST_RESOURCES_DIR / p


def read_test_resource(p: Path | str) -> str:
    test_resource = get_test_resource_path(p)
    assert test_resource.is_file(), f"Test resources: {test_resource} does not exist."
    return test_resource.read_text()
