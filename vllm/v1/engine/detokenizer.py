# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import Optional

import tokenizers
from packaging import version
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from transformers import PreTrainedTokenizerFast

from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.logger import init_logger
from vllm.transformers_utils.detokenizer_utils import (
    AnyTokenizer, convert_prompt_ids_to_tokens, detokenize_incrementally)
from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)

# Only tokenizers >= 0.21.1 supports DecodeStream used for
# FastIncrementalDetokenizer.
USE_FAST_DETOKENIZER = version.parse(
    tokenizers.__version__) >= version.parse("0.21.1")

# Error string from https://github.com/huggingface/tokenizers/blob/909fdde2a4ffedd9295206f705eb612be2a91b12/tokenizers/src/tokenizer/mod.rs#L1042
INVALID_PREFIX_ERR_MSG = "Invalid prefix encountered"


class IncrementalDetokenizer:

    def __init__(self):
        self.token_ids: list[int] = []

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids

    def update(self, new_token_ids: list[int],
               stop_terminated: bool) -> Optional[str]:
        self.token_ids.extend(new_token_ids)
        return None

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        return ""

    @classmethod
    def from_new_request(
        cls,
        tokenizer: Optional[AnyTokenizer],
        request: EngineCoreRequest,
    ) -> "IncrementalDetokenizer":

        assert request.sampling_params is not None

        if tokenizer is None:
            # No tokenizer => skipping detokenization.
            return IncrementalDetokenizer()

        if USE_FAST_DETOKENIZER and isinstance(tokenizer,
                                               PreTrainedTokenizerFast):
            # Fast tokenizer => use tokenizers library DecodeStream.
            return FastIncrementalDetokenizer(tokenizer, request)

        # Fall back to slow python-based incremental detokenization.
        return SlowIncrementalDetokenizer(tokenizer, request)


class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):

    def __init__(self, request: EngineCoreRequest):
        super().__init__()

        # Stop strings
        params = request.sampling_params
        assert params is not None
        self.stop = stop = params.stop
        self.min_tokens = params.min_tokens
        self.include_stop_str_in_output = params.include_stop_str_in_output

        # Number of chars to hold back when stop strings are to be excluded
        # from streamed output.
        if stop and not self.include_stop_str_in_output:
            self.stop_buffer_length = max(len(s) for s in stop) - 1
        else:
            self.stop_buffer_length = 0
        self._last_output_text_offset: int = 0

        # Generation data
        self.output_text = ""

    def update(self, new_token_ids: list[int],
               stop_terminated: bool) -> Optional[str]:
        """
        Update RequestState for the request_id by:
            1) Detokenize all token ids at once (batch detokenization).
            2) Evaluate stop criteria.

        Return matched stop string or None.
        """
        if not new_token_ids:
            # Skip detokenization if no new token ids.
            return None

        if stop_terminated and not self.include_stop_str_in_output:
            # If stop-terminated, exclude last token from detokenization
            # based on include_stop_str_in_output parameter.
            skipped_stop_token_id = new_token_ids[-1]
            new_token_ids = new_token_ids[:-1]
        else:
            skipped_stop_token_id = None

        # 1) Add new tokens and detokenize all at once (batch mode).
        stop_check_offset = len(self.output_text)
        self.token_ids.extend(new_token_ids)
        
        # Decode all output tokens at once instead of incrementally
        self.output_text = self.decode_all(self.output_token_ids)

        if skipped_stop_token_id is not None:
            # Cleanup after skipping detokenization.
            self.token_ids.append(skipped_stop_token_id)

        # 2) Evaluate stop strings.
        stop_string = None
        if self.stop and len(self.output_token_ids) > self.min_tokens:
            stop = StopChecker.check_stop_strings(
                output_text=self.output_text,
                new_char_count=len(self.output_text) - stop_check_offset,
                stop=self.stop,
                include_in_output=self.include_stop_str_in_output,
            )
            if stop is not None:
                stop_string, truncate_to = stop
                if truncate_to != -1:
                    self.output_text = self.output_text[:truncate_to]

        return stop_string

    @abstractmethod
    def decode_all(self, token_ids: list[int]) -> str:
        raise NotImplementedError

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        """If delta is True, only new text since the last call to
        this method is returned"""

        # We return the full output text if the sequence is finished.
        buffer_length = 0 if finished else self.stop_buffer_length
        if not delta:
            return self.output_text[:-buffer_length] if buffer_length else (
                self.output_text)
        length = len(self.output_text) - buffer_length
        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return self.output_text[last_offset:length]
        return ""


class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):

    def __init__(self, tokenizer: PreTrainedTokenizerFast,
                 request: EngineCoreRequest):
        super().__init__(request)

        sampling_params = request.sampling_params
        assert sampling_params is not None

        self.request_id = request.request_id
        self.skip_special_tokens = sampling_params.skip_special_tokens
        self.tokenizer: Tokenizer = tokenizer._tokenizer
        self.spaces_between_special_tokens = (
            sampling_params.skip_special_tokens
            or sampling_params.spaces_between_special_tokens)

    def decode_all(self, token_ids: list[int]) -> str:
        """Decode all tokens at once using batch detokenization."""
        try:
            text = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=self.skip_special_tokens
            )
            return text
        except Exception as e:
            logger.exception(
                "Encountered error during batch detokenization for request %s: %s",
                self.request_id, str(e))
            # Return empty string on error
            return ""


class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):

    def __init__(self, tokenizer: AnyTokenizer, request: EngineCoreRequest):
        super().__init__(request)

        self.tokenizer = tokenizer
        params = request.sampling_params
        assert params is not None

        self.token_ids.extend(request.prompt_token_ids)
        self.prompt_len = len(request.prompt_token_ids)

        self.skip_special_tokens = params.skip_special_tokens
        self.spaces_between_special_tokens = (
            params.spaces_between_special_tokens)

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids if not self.prompt_len else (
            self.token_ids[self.prompt_len:])

    def decode_all(self, token_ids: list[int]) -> str:
        """Decode all tokens at once using batch detokenization."""
        try:
            text = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=self.skip_special_tokens
            )
            return text
        except Exception as e:
            logger.exception(
                "Encountered error during batch detokenization: %s", str(e))
            # Return empty string on error
            return ""
