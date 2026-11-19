import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from native_timing_inputs import check, normalized_messages, tokenize


def test_text_parts_keep_http_newline_join_without_mutating_requests():
    messages = [dict(role="user", content=[dict(type="text", text="first"), dict(type="text", text="second")])]
    assert normalized_messages(messages) == [dict(role="user", content="first\nsecond")]
    assert isinstance(messages[0]["content"], list)


def test_image_inputs_are_not_silently_dropped_from_native_workload():
    with pytest.raises(ValueError, match="text parts"):
        normalized_messages([dict(role="user", content=[dict(type="image_url", image_url={"url": "fixture"})])])


def test_native_chat_format_and_tokenizer_bos_are_explicit():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=True)
            return "formatted"

        def encode(self, text, **kwargs):
            assert text == "formatted" and kwargs == dict(add_special_tokens=False)
            return [1, 2, 3]

    assert tokenize(Tokenizer(), [dict(role="user", content="text")]) == [1, 2, 3]


def test_equal_length_but_different_prompt_ids_fail_before_model_submission():
    expected = dict(count=3, token_ids_sha256=hashlib.sha256(json.dumps([1, 2, 3]).encode()).hexdigest())
    check([1, 2, 3], expected)
    with pytest.raises(ValueError, match="original HTTP population"):
        check([1, 2, 4], expected)
