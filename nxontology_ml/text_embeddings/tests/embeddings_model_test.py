from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from nxontology import NXOntology
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.utils import ModelOutput

from nxontology_ml.data import read_training_data
from nxontology_ml.text_embeddings.embeddings_model import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_SIZES,
    AutoModelEmbeddings,
    _cache_filename,
    _LazyAutoModel,
)


def test_embed_node(nxo: NXOntology[str], embeddings_cache_dir: Path) -> None:
    ame = AutoModelEmbeddings.from_pretrained(
        pretrained_model_name=DEFAULT_EMBEDDING_MODEL,
        cache_dir=embeddings_cache_dir,
    )
    X, _ = read_training_data(nxo=nxo, take=10)
    vecs = np.array([ame.embed_node(nxo.node_info(node_id)) for node_id in X])
    assert vecs.shape == (10, EMBEDDING_SIZES[DEFAULT_EMBEDDING_MODEL])
    # Make sure no network calls were made
    assert dict(ame._counter) == {"AutoModelEmbeddings/CACHE_HIT": 10}


def test_caching(nxo: NXOntology[str], tmp_path: Path) -> None:
    # Mock
    class AutoModelMock(_LazyAutoModel):
        @property
        def tokenizer(self) -> PreTrainedTokenizerBase:
            if self._tokenizer is None:
                mock = Mock(spec=PreTrainedTokenizerBase)
                mock.return_value = {}
                self._tokenizer = mock  # type: ignore[assignment]
            return self._tokenizer

        @property
        def model(self) -> PreTrainedModel:
            if self._model is None:
                mock = Mock(spec=PreTrainedModel)
                model_output = Mock(spec=ModelOutput)
                model_output.pooler_output = torch.Tensor([42])
                mock.return_value = model_output
                self._model = mock  # type: ignore[assignment]
            return self._model

    # Actual test
    model_mock = AutoModelMock(DEFAULT_EMBEDDING_MODEL)
    ame = AutoModelEmbeddings.from_pretrained(
        pretrained_model_name=DEFAULT_EMBEDDING_MODEL,
        lazy_model=model_mock,
        cache_dir=tmp_path,
    )
    test_node = "DOID:0050890"
    vec = ame.embed_node(nxo.node_info(test_node))
    assert np.equal(vec, np.array([42]))
    assert dict(ame._counter) == {"AutoModelEmbeddings/CACHE_MISS": 1}
    vec2 = ame.embed_node(nxo.node_info(test_node))
    assert np.equal(vec2, np.array([42]))
    assert dict(ame._counter) == {
        "AutoModelEmbeddings/CACHE_MISS": 1,
        "AutoModelEmbeddings/CACHE_HIT": 1,
    }


def test_lazy_automodel() -> None:
    model_name = "test_model_name"
    tokenizer_cls_mock = Mock(spec=PreTrainedTokenizerBase)
    model_cls_mock = Mock(spec=PreTrainedModel)
    lam = _LazyAutoModel(
        pretrained_model_name=model_name,
        tokenizer_cls=tokenizer_cls_mock,
        model_cls=model_cls_mock,
    )
    # Model
    model_cls_mock.from_pretrained.assert_not_called()
    _ = lam.model
    model_cls_mock.from_pretrained.assert_called_once_with(model_name)
    _ = lam.model  # Cached
    model_cls_mock.from_pretrained.assert_called_once_with(model_name)

    # Tokenizer
    tokenizer_cls_mock.from_pretrained.assert_not_called()
    _ = lam.tokenizer
    tokenizer_cls_mock.from_pretrained.assert_called_once_with(model_name)
    _ = lam.tokenizer  # Cached
    tokenizer_cls_mock.from_pretrained.assert_called_once_with(model_name)


def test_cache_filename() -> None:
    fn = _cache_filename(pretrained_model_name=DEFAULT_EMBEDDING_MODEL)
    assert fn == "michiyasunaga_BioLinkBERT_base.ldb"


@pytest.mark.skip(reason="Pulls resources off the internet")
def test_builder(tmp_path: Path) -> None:
    ame = AutoModelEmbeddings.from_pretrained(
        pretrained_model_name=DEFAULT_EMBEDDING_MODEL,
        # Ensure that cache doesn't exit
        cache_dir=tmp_path,
    )
    vec = ame.embed_text("Sunitinib is a tyrosine kinase inhibitor")
    assert vec.shape == (EMBEDDING_SIZES[DEFAULT_EMBEDDING_MODEL],)
    assert dict(ame._counter) == {"AutoModelEmbeddings/CACHE_MISS": 1}
