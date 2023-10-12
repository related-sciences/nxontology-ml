import logging
import pickle
import re
from collections import Counter
from hashlib import sha1
from pathlib import Path

import numpy as np
from nxontology.node import NodeInfo
from torch import Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import ModelOutput

from nxontology_ml.gpt_tagger._cache import LazyLSM
from nxontology_ml.utils import ROOT_DIR

DEFAULT_EMBEDDING_MODEL = "michiyasunaga/BioLinkBERT-base"
EMBEDDING_SIZES: dict[str, int] = {DEFAULT_EMBEDDING_MODEL: 768}
_model_poolers: dict[str, str] = {DEFAULT_EMBEDDING_MODEL: "pooler_output"}


def _cache_path(pretrained_model_name: str) -> Path:
    safe_prefix = re.sub("[^0-9a-zA-Z]+", "_", pretrained_model_name)
    return ROOT_DIR / f".cache/{safe_prefix}.ldb"


class _LazyAutoModel:
    """
    Avoid pulling the PreTrainedModel resources from hugging face until actually needed
    (Embeddings are cached so we mostly don't actually need the model)
    """

    def __init__(
        self,
        pretrained_model_name: str,
        tokenizer_cls: PreTrainedTokenizerBase = AutoTokenizer,
        model_cls: PreTrainedModel = AutoModel,
    ):
        self._pretrained_model_name = pretrained_model_name
        self._tokenizer_cls = tokenizer_cls
        self._model_cls = model_cls
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            self._tokenizer = self._tokenizer_cls.from_pretrained(
                self._pretrained_model_name
            )
        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            self._model = self._model_cls.from_pretrained(self._pretrained_model_name)
        return self._model


class AutoModelEmbeddings:
    """
    Simple wrapper around Automodels from huggingface/transformers
    """

    def __init__(
        self,
        lazy_model: _LazyAutoModel,
        pooler_attr: str,
        cache: LazyLSM,
        counter: Counter[str],
    ):
        self._lazy_model = lazy_model
        self._pooler_attr = pooler_attr
        self._cache = cache
        self._counter = counter

    def embed_node(self, node: NodeInfo[str]) -> np.array:
        text = f"{node.data['efo_label']}: {node.data['efo_definition']}"
        return self.embed_text(text)

    def embed_text(self, text: str) -> np.ndarray:
        # Note: We could cache keyed on node id, but text seems safer (albeit less space efficient)
        cache_key = sha1(text.encode()).hexdigest()
        if cache_key in self._cache:
            self._counter[f"{self.__class__.__name__}/CACHE_HIT"] += 1
            return pickle.loads(self._cache[cache_key])
        self._counter[f"{self.__class__.__name__}/CACHE_MISS"] += 1
        # Note: Only works with pytorch return_tensors ATM
        inputs: BatchEncoding = self._lazy_model.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        outputs: ModelOutput = self._lazy_model.model(**inputs)
        assert hasattr(
            outputs, self._pooler_attr
        ), f"Model output should have a '{self._pooler_attr}' attribute: {outputs}"
        tensor = getattr(outputs, self._pooler_attr)
        assert isinstance(tensor, Tensor)
        assert tensor.shape[0] == 1  # Batch size is always 1
        vector = tensor[0].detach().numpy()
        self._cache[cache_key] = pickle.dumps(vector)
        return vector

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: str,
        cache_path: Path | None = None,
        lazy_model: _LazyAutoModel | None = None,
        counter: Counter[str] | None = None,
    ) -> "AutoModelEmbeddings":
        """
        Note: pretrained_model_name should be an encoder only model (e.g. BERT)
        """
        # FIXME: should we add truncation of input??
        cache_filename = (cache_path or _cache_path(pretrained_model_name)).as_posix()
        logging.debug(f"Caching embeddings into: {cache_filename}")
        return cls(
            lazy_model=lazy_model or _LazyAutoModel(pretrained_model_name),
            pooler_attr=_model_poolers[pretrained_model_name],
            cache=LazyLSM(filename=cache_filename),
            counter=counter or Counter(),
        )
