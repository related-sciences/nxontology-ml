import inspect
from collections.abc import Iterable
from io import StringIO
from pathlib import Path

import pandas as pd
from nxontology.node import NodeInfo
from pandas._testing import assert_frame_equal
from pandas._typing import DtypeArg

from nxontology_ml.data import get_efo_otar_slim


def _caller_path(caller_path: Path | None, stack_lvl: int = 2) -> Path:
    # Allows to "magically" resolve the test resources next to the caller test script
    # stack_lvl=2 => This function is called by a function called by test (2 levels down)
    return caller_path or Path(inspect.stack()[stack_lvl][1]).parent


def get_test_resource_path(
    p: Path | str,
    caller_path: Path | None = None,
) -> Path:
    if isinstance(p, str):
        p = _caller_path(caller_path) / "test_resources" / p
    return p


def read_test_resource(
    p: Path | str,
    caller_path: Path | None = None,
) -> str:
    test_resource = get_test_resource_path(p, _caller_path(caller_path))
    assert test_resource.is_file(), f"Test resources: {test_resource} does not exist."
    return test_resource.read_text()


def read_test_dataframe(
    p: Path | str,
    orient: str = "records",
    dtype: DtypeArg | None = None,
    caller_path: Path | None = None,
) -> pd.DataFrame:
    test_resource = get_test_resource_path(p, _caller_path(caller_path))
    assert test_resource.is_file(), f"Test resources: {test_resource} does not exist."
    # Note: These resources are usually saved with `df.to_json(orient="records")`
    return pd.read_json(
        StringIO(test_resource.read_text()),
        orient=orient,
        dtype=dtype,
    )


def assert_frame_equal_to(
    df: pd.DataFrame,
    p: Path | str,
    orient: str = "records",
    dtype: DtypeArg | None = None,
    check_dtype: bool = False,
    caller_path: Path | None = None,
) -> None:
    # Note: pd dtypes are finicky when serializing so we skip type checking by default
    # (In most cases the value difference will capture the behavior we want to test)
    assert_frame_equal(
        df,
        read_test_dataframe(p, orient, dtype, _caller_path(caller_path)),
        check_dtype,
    )


def get_test_nodes() -> Iterable[NodeInfo[str]]:
    test_efo_otar = get_test_resource_path("sampled_efo_otar_slim.json")
    nxo = get_efo_otar_slim(url=test_efo_otar.as_uri())
    yield from (nxo.node_info(node) for node in sorted(nxo.graph))
