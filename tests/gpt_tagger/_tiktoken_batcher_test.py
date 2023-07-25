from copy import copy
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import tiktoken
from tiktoken import Encoding

from nxontology_ml.gpt_tagger._openai_models import _4K
from nxontology_ml.gpt_tagger._tiktoken_batcher import _TiktokenBatcher
from tests.gpt_tagger._utils import precision_config
from tests.utils import get_test_resource_path


@pytest.fixture
def tiktoken_cl100k_encoding() -> Encoding:
    return tiktoken.get_encoding("cl100k_base")


def test_add_tokens(tiktoken_cl100k_encoding: Encoding) -> None:
    record = "record"
    record_len = len(tiktoken_cl100k_encoding.encode(record))

    batcher = _TiktokenBatcher(
        max_token_cnt=2 * record_len, tiktoken_encoding=tiktoken_cl100k_encoding
    )
    assert batcher._record_buffer == []
    assert batcher._buffer_token_cnt == 0

    ret = batcher.add_record_to_buffer(record)
    assert ret is None
    assert batcher._record_buffer == [record]
    assert batcher._buffer_token_cnt == record_len

    ret = batcher.add_record_to_buffer(record)
    assert ret is None
    assert batcher._record_buffer == [record, record]
    assert batcher._buffer_token_cnt == 2 * record_len

    ret = batcher.add_record_to_buffer(record)
    assert ret == [record, record]
    assert batcher._record_buffer == [record]
    assert batcher._buffer_token_cnt == record_len

    ret = batcher.flush_buffer()
    assert ret == [record]
    assert batcher._record_buffer == []
    assert batcher._buffer_token_cnt == 0

    # Error triggered by internal tempering
    batcher._do_add_record_to_buffer(record)
    batcher._do_add_record_to_buffer(record)
    with pytest.raises(ValueError, match="Buffer size exceeded"):
        batcher._do_add_record_to_buffer(record)


def test_from_config(tiktoken_cl100k_encoding: Encoding) -> None:
    # Valid config
    batcher = _TiktokenBatcher.from_config(precision_config)
    assert batcher._tiktoken_encoding == tiktoken_cl100k_encoding

    prompt = get_test_resource_path("precision_v1.txt")
    prompt_tokens = len(tiktoken_cl100k_encoding.encode(prompt.read_text()))
    assert prompt_tokens > 0
    assert batcher._max_token_cnt == _4K - prompt_tokens

    # Prompt too long
    invalid_test_config = copy(precision_config)
    with NamedTemporaryFile() as tmpfile:
        # Mk faulty prompt content
        prompt_path = Path(tmpfile.name)
        # Uncommon symbol has >= 1 token per char
        prompt_content = "Ⓡ" * (_4K + 1)
        assert len(tiktoken_cl100k_encoding.encode(prompt_content)) >= _4K + 1
        prompt_path.write_text(prompt_content)

        invalid_test_config.prompt_path = prompt_path
        with pytest.raises(
            ValueError,
            match="The provided prompt has more tokens than the window of the model.",
        ):
            _TiktokenBatcher.from_config(invalid_test_config)
