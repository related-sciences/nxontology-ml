import hashlib
from collections import Counter
from collections.abc import Iterator, MutableMapping
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, ParamSpecKwargs

from lsm import LSM

from nxontology_ml.gpt_tagger._models import TaskConfig
from nxontology_ml.gpt_tagger._utils import config_to_cache_namespace, counter_or_empty
from nxontology_ml.utils import ROOT_DIR


class _Cache:
    """
    Helper class to cache nodes' labels (to disk by default)
    The key is based on the node's features. The value is its label.

    Notes:
        - The cache is at the node level (not the prompt level)
        - A mandatory namespace is used for cache invalidation
        - Deletion of old namespaces is to be done manually by the user
        - Keys can be optionally hashed if a hash fn is provided
    """

    def __init__(
        self,
        storage: MutableMapping[str, bytes],
        counter: Counter[str],
        namespace: str = "",
        key_hash_fn: str | None = None,
    ):
        """
        Intended to get constructed using cls.from_config(config)
        """
        self._storage = storage
        self._counter = counter
        self._namespace = namespace
        self._key_hash_fn = key_hash_fn

    def get(self, key: str, default: str | None = None) -> str | None:
        self._counter["Cache/get"] += 1
        key = self._mk_key(key)
        if key not in self._storage:
            return default
        val: str | bytes = self._storage[key]
        if isinstance(val, bytes):
            val = val.decode()
        return val

    def __setitem__(self, key: str, value: str) -> None:
        self._counter["Cache/set"] += 1
        self._storage[self._mk_key(key)] = value.encode()

    def __delitem__(self, key: str) -> None:
        self._counter["Cache/del"] += 1
        del self._storage[self._mk_key(key)]

    def _mk_key(self, key: str) -> str:
        if self._key_hash_fn:
            h = hashlib.new(self._key_hash_fn, usedforsecurity=False)
            h.update(key.encode())
            key = h.hexdigest()
        return f"{self._namespace}/{key}"

    @classmethod
    def from_config(
        cls,
        config: TaskConfig,
        counter: Counter[str] | None = None,
        cache_path: Path | None = None,
    ) -> "_Cache":
        cache_namespace = config_to_cache_namespace(config)
        if not cache_path:
            cache_path = ROOT_DIR / f".cache/{cache_namespace}.ldb"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(
            storage=LazyLSM(cache_path.as_posix()),
            namespace="",  # Namespace is already part of the storage path
            key_hash_fn=config.cache_key_hash_fn,
            counter=counter_or_empty(counter),
        )


class LazyLSM(MutableMapping[str, bytes], AbstractContextManager[Any]):
    """
    Thin wrapper around SQLite4's LSM, so that:
    - LSM can be opened lazily (avoid creating the db file when simply instantiating LSM)
    - LSM follows python's MutableMapping & AbstractContextManager protocols

    Note: LSM has an open_database parameter but it appears broken
    """

    def __init__(self, filename: str, **kwargs: ParamSpecKwargs):
        self._lsm: LSM | None = None
        self._filename = filename
        self._lsm_kwargs = kwargs

    def __open_db(self) -> None:
        if not self._lsm:
            self._lsm = LSM(filename=self._filename, **self._lsm_kwargs)

    def __setitem__(self, __k: str, __v: bytes) -> None:
        self.__open_db()
        assert self._lsm
        self._lsm.__setitem__(__k, __v)

    def __delitem__(self, __v: str) -> None:
        self.__open_db()
        assert self._lsm
        self._lsm.__delitem__(__v)

    def __getitem__(self, __k: str) -> bytes:
        self.__open_db()
        return self._lsm.__getitem__(__k)  # type: ignore

    def __iter__(self) -> Iterator[str]:
        self.__open_db()
        return self._lsm.__iter__()  # type: ignore

    def __len__(self) -> int:
        if not Path(self._filename).exists():
            return 0
        self.__open_db()
        return sum(1 for _ in iter(self))

    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> None:
        if self._lsm:
            self._lsm.__exit__(__exc_type, __exc_value, __traceback)
