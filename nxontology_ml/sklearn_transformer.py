from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
from nxontology.node import NodeInfo
from pandas.core.dtypes.base import ExtensionDtype
from sklearn.base import TransformerMixin


@dataclass
class NodeFeatures:
    nodes: list[NodeInfo[str]]
    num_features: pd.DataFrame
    cat_features: pd.DataFrame

    @classmethod
    def from_nodes(cls, nodes: list[NodeInfo[str]]) -> "NodeFeatures":
        return cls(
            nodes=nodes,
            num_features=pd.DataFrame(dtype=np.float32),
            cat_features=pd.DataFrame(dtype=object),
        )

    def __len__(self) -> int:
        return len(self.nodes)


T = TypeVar("T")
U = TypeVar("U")


class NoFitTransformer(TransformerMixin, Generic[T, U], ABC):  # type: ignore[misc]
    """
    Transformer with a no-op fit function
    """

    def fit(
        self, X: T, y: Iterable[str] | None = None, **fit_params: Any
    ) -> "NoFitTransformer[T, U]":
        return self  # Noop

    @abstractmethod
    def transform(self, X: T, copy: bool | None = None) -> U:
        raise NotImplementedError()  # pragma: no cover


class DataFrameFnTransformer(NoFitTransformer[NodeFeatures, NodeFeatures]):
    """
    Applies a function to the input and create new feature
    """

    def __init__(
        self,
        num_features_fn: Callable[[NodeInfo[str]], np.array] | None = None,
        num_features_names: list[str] | None = None,
        num_feature_dtype: ExtensionDtype | None = None,
        cat_features_fn: Callable[[NodeInfo[str]], np.array] | None = None,
        cat_features_names: list[str] | None = None,
        enabled: bool = True,
    ):
        self._num_features_fn = num_features_fn
        self._num_features_names = num_features_names
        self._num_feature_dtype = num_feature_dtype
        self._cat_features_fn = cat_features_fn
        self._cat_features_names = cat_features_names
        self._enabled = enabled

    def transform(self, X: NodeFeatures, copy: bool | None = None) -> NodeFeatures:
        if not self._enabled:
            return X
        if self._num_features_fn:
            assert self._num_features_names
            new_features = pd.DataFrame(
                data=[self._num_features_fn(node) for node in X.nodes],
                columns=self._num_features_names,
                dtype=self._num_feature_dtype,
            )
            X.num_features = pd.concat([X.num_features, new_features], axis=1)
        if self._cat_features_fn:
            assert self._cat_features_names
            new_features = pd.DataFrame(
                data=[self._cat_features_fn(node) for node in X.nodes],
                columns=self._cat_features_names,
            )
            X.cat_features = pd.concat([X.cat_features, new_features], axis=1)
        return X
