from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import islice
from typing import TypeVar

import numpy as np
from faiss import Index, index_factory

from nxontology_ml.text_embeddings.embeddings_model import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_SIZES,
)


@dataclass
class NodeIdLabel:
    node_id: str
    label: str


@dataclass
class LabeledNodeVec:
    node_id_label: NodeIdLabel
    vec: np.array


@dataclass
class NeighborDist:
    node_id_label: NodeIdLabel
    dist: float


T = TypeVar("T")


def _batched(iterable: Iterable[T], n: int) -> Iterable[Iterable[T]]:
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    # Src: https://docs.python.org/3/library/itertools.html
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class FaissIndex:
    """
    Thin wrapper on top of Meta's Faiss to index *labelled* vectors
    """

    def __init__(self, index: Index, metadata_kv: dict[int, NodeIdLabel]):
        self._index = index
        self._metadata_kv = metadata_kv

    def search(self, vector: np.array, k: int) -> Iterable[NeighborDist]:
        # For now, we only query one vector a the time
        assert vector.shape == (self._index.d,)
        nns_dist, nns_idx = self._index.search(vector.reshape((1, self._index.d)), k)
        assert len(nns_dist) == len(nns_idx) == 1
        for dist, idx in sorted(
            zip(nns_dist[0], nns_idx[0], strict=True), reverse=True
        ):
            if idx == -1:
                # Faiss pads missing neighbors with -1
                continue
            yield NeighborDist(dist=dist, node_id_label=self._metadata_kv[idx])

    @property
    def ntotal(self) -> int:
        n = self._index.ntotal
        assert isinstance(n, int)
        return n


REPORTHOOK_FN = Callable[[int], None]

DEFAULT_FAISS_INDEX = "HNSW"


class FaissIndexBuilder:
    def __init__(
        self,
        index: Index,
        index_name: str,
        metadata_kv: dict[int, NodeIdLabel] | None = None,
    ):
        self._index = index
        self._index_name = index_name
        self._metadata_kv = metadata_kv or {}
        self._node_cnt = 0

    def add_nodes(
        self, node_vecs: Iterable[LabeledNodeVec], batch_size: int = 32
    ) -> None:
        for batch in _batched(node_vecs, batch_size):
            vecs = []
            for node in batch:
                vecs.append(node.vec)
                # Faiss uses node index as identifier
                self._metadata_kv[self._node_cnt] = node.node_id_label
                self._node_cnt += 1
            self._index.add(np.vstack(vecs))

    def build_index(self) -> FaissIndex:
        return FaissIndex(self._index, self._metadata_kv)

    @classmethod
    def from_index_name(
        cls,
        index_name: str = DEFAULT_FAISS_INDEX,
        dim: int = EMBEDDING_SIZES[DEFAULT_EMBEDDING_MODEL],
    ) -> "FaissIndexBuilder":
        # For indexes, take a look at:
        # - https://github.com/facebookresearch/faiss/wiki/The-index-factory
        # - https://www.pinecone.io/learn/series/faiss/vector-indexes/
        return cls(
            index=index_factory(dim, index_name),
            index_name=index_name,
        )
