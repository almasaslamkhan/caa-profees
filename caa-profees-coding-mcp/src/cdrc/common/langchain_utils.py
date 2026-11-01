import asyncio
import json
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, Generator, List, Optional, Union

import openai
from langchain_community.callbacks.manager import openai_callback_var
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import ChatResult, Generation, LLMResult
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI
from loguru import logger

DEFAULT_CORRECTION_PROMPT = """
When trying to parse the following text, it couldn't be parsed by the following schema. Please fix the text to be parsable by the schema.
### Wrongly Formatted Text:\n\n{text}\n\n
### Desired Schema:\n{format_instructions}\n\n
### Parsing error:\n{parsing_error}\n\n
DO NOT ADD any content before or after the JSON. The output should be a valid JSON object that can be directly parsed by the schema."""

TRANSIENT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class PydanticOutputParserWithRetries(PydanticOutputParser):
    correction_model: BaseChatModel
    num_retries: int = 3
    correction_prompt: str = DEFAULT_CORRECTION_PROMPT

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        try:
            return super().parse_result(result, partial=partial)
        except OutputParserException as e:
            messages_: List[BaseMessage] = []
            res_copy = deepcopy(result)
            text = res_copy[0].text
            messages_.append(
                HumanMessage(
                    content=self.correction_prompt.format(text=text, format_instructions=self.get_format_instructions(), parsing_error=str(e))
                )
            )
            for _ in range(self.num_retries):
                try:
                    response = self.correction_model.invoke(messages_)
                    messages_.append(response)
                    res_copy[0].text = response.content
                    return super().parse_result(res_copy, partial=partial)
                except OutputParserException as e:
                    messages_.append(
                        HumanMessage(
                            content=f"The latest output still couldn't be parsed into the given schema. Please try again. Parsing error: {str(e)}"
                        )
                    )
            logger.warning(f"Failed to parse the output after {self.num_retries} retries. {result[0].text}")
            return super().parse_result(result, partial=partial)

    def parse(self, text: str) -> Any:
        # try to parse the text, if not successful, pass the text to the correction chain and try to parse the output using the same parser
        try:
            return super().parse(text)
        except OutputParserException:
            messages_: List[BaseMessage] = []
            messages_.append(HumanMessage(content=self.correction_prompt.format(text=text, format_instructions=self.get_format_instructions())))
            for _ in range(self.num_retries):
                response = self.correction_model.invoke(messages_)
                messages_.append(response)
                try:
                    return super().parse(response.content)
                except OutputParserException:
                    messages_.append(HumanMessage(content="The output still couldn't be parsed into the given schema. Please try again."))
            logger.warning(f"Failed to parse the output after {self.num_retries} retries. Returning the original text.")
            return super().parse(text)


def extract_retry_seconds(error: openai.RateLimitError) -> Optional[float]:
    import re

    try:
        error_str = str(error)
        if match := re.search(r"retry after (\d+) seconds", error_str.lower()):
            return float(match.group(1))
        if match := re.search(r"(\d+) seconds", error_str.lower()):
            return float(match.group(1))
        return None
    except Exception:
        return None


def get_retry_delay(attempt: int, error: Optional[openai.RateLimitError] = None, retry_delay: float = 1.0) -> float:
    """Implements smart delay with fallback to exponential backoff"""
    if error:
        # Try to get the retry time from the error message
        if suggested_delay := extract_retry_seconds(error):
            return suggested_delay + 1  # Add 1 second buffer

    # Fallback to exponential backoff
    return min(retry_delay * (2**attempt), 120)  # Cap at 2 minutes


def _extract_http_error_details(exc: Exception) -> Dict[str, Any]:
    """Best-effort extraction of HTTP status/body attached by OpenAI/LangChain."""
    response = getattr(exc, "response", None)
    if response is None:
        return {}

    details: Dict[str, Any] = {}

    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        details["status_code"] = status_code

    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            details["content_type"] = headers.get("content-type")
        except Exception:
            pass

    text = None
    try:
        text = response.text
    except Exception:
        pass

    if not text:
        try:
            content = getattr(response, "content", None)
            if isinstance(content, bytes):
                text = content.decode("utf-8", errors="replace")
        except Exception:
            pass

    if text:
        text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                details["json_keys"] = sorted(parsed.keys())
                if "error" in parsed:
                    details["error"] = parsed["error"]
                details["body_preview"] = json.dumps(parsed, ensure_ascii=True)[:1000]
            else:
                details["body_preview"] = text[:1000]
        except Exception:
            details["body_preview"] = text[:1000]

    return details


def _is_malformed_chat_response_error(exc: Exception) -> bool:
    """Identify LangChain/OpenAI-compatible responses that are missing chat completions fields."""
    message = str(exc)
    if "choices" in message and ("Response missing" in message or "null value for 'choices'" in message):
        return True

    details = _extract_http_error_details(exc)
    json_keys = details.get("json_keys")
    if isinstance(json_keys, list) and "choices" not in json_keys:
        return True

    return False


class ChatOpenAIWithRetries(ChatOpenAI):
    """ChatOpenAI wrapper that implements infinite retries for rate limit errors and not found errors"""

    max_retries_on_rate_limit: int = 5  # -1 means infinite retries
    max_retries_on_not_found: int = 120
    max_retries_on_timeout: int = 6  # Number of times to retry on timeout errors
    max_retries_on_malformed_response: int = 3
    retry_delay: float = 1.0  # Base delay between retries in seconds
    timeout: float = 600  # Request timeout in seconds

    # override the default generate method to add rate limit retries
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        attempt_of_rate_limit = 0
        attempt_of_not_found = 0
        attempt_of_timeout = 0
        attempt_of_malformed_response = 0
        current_timeout = self.timeout

        while True:
            try:
                return super()._generate(messages, stop, run_manager, timeout=current_timeout, **kwargs)
            except openai.RateLimitError as e:
                attempt_of_rate_limit += 1
                if self.max_retries_on_rate_limit < 0 or attempt_of_rate_limit <= self.max_retries_on_rate_limit:
                    delay = get_retry_delay(attempt_of_rate_limit, error=e, retry_delay=self.retry_delay)
                    logger.warning(
                        f"Rate limit error, retrying in {delay} seconds... "
                        f"attempt: {attempt_of_rate_limit} of "
                        f"{self.max_retries_on_rate_limit if self.max_retries_on_rate_limit >= 0 else 'infinite'}"
                    )
                    if run_manager:
                        run_manager.on_retry(
                            error=e, retry_state={"attempt": attempt_of_rate_limit, "delay": delay, "extracted_delay": extract_retry_seconds(e)}
                        )
                    time.sleep(delay)
                    continue
                else:
                    raise e
            except openai.NotFoundError as e:
                attempt_of_not_found += 1
                if self.max_retries_on_not_found < 0 or attempt_of_not_found <= self.max_retries_on_not_found:
                    logger.warning(
                        f"Model not found error, possibly due to APIM, retrying in {self.retry_delay * 10} seconds... "
                        f"attempt: {attempt_of_not_found} of "
                        f"{self.max_retries_on_not_found if self.max_retries_on_not_found >= 0 else 'infinite'}"
                    )
                    if run_manager:
                        run_manager.on_retry(
                            error=e,
                            retry_state={"attempt": attempt_of_not_found, "delay": self.retry_delay * 10, "extracted_delay": self.retry_delay * 10},
                        )
                    time.sleep(self.retry_delay * 10)
                    continue
                else:
                    raise e
            except (openai.APIConnectionError, TimeoutError) as e:
                attempt_of_timeout += 1
                if self.max_retries_on_timeout < 0 or attempt_of_timeout <= self.max_retries_on_timeout:
                    delay = self.retry_delay * 2
                    logger.warning(
                        f"Timeout error, retrying in {delay} seconds... "
                        f"attempt: {attempt_of_timeout} of "
                        f"{self.max_retries_on_timeout if self.max_retries_on_timeout >= 0 else 'infinite'}; "
                        f"next timeout: {current_timeout * 2:.1f}s"
                    )
                    if run_manager:
                        run_manager.on_retry(error=e, retry_state={"attempt": attempt_of_timeout, "delay": delay, "extracted_delay": delay})
                    time.sleep(delay)
                    current_timeout *= 3  # 3*timeout for next retry
                    continue
                else:
                    raise e
            except (KeyError, TypeError, ValueError) as e:
                if not _is_malformed_chat_response_error(e):
                    raise e

                attempt_of_malformed_response += 1
                details = _extract_http_error_details(e)
                if self.max_retries_on_malformed_response < 0 or attempt_of_malformed_response <= self.max_retries_on_malformed_response:
                    delay = self.retry_delay * 2
                    logger.warning(
                        "Malformed chat-completions response, retrying in {} seconds... attempt: {} of {}. Details: {}",
                        delay,
                        attempt_of_malformed_response,
                        self.max_retries_on_malformed_response if self.max_retries_on_malformed_response >= 0 else "infinite",
                        details or {"error": str(e)},
                    )
                    if run_manager:
                        run_manager.on_retry(
                            error=e,
                            retry_state={"attempt": attempt_of_malformed_response, "delay": delay, "details": details or {"error": str(e)}},
                        )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        "Malformed chat-completions response after retries exhausted. Details: {}",
                        details or {"error": str(e)},
                    )
                    raise e

    # override the default async generate method to add rate limit retries
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        attempt = 0
        attempt_on_not_found = 0
        attempt_of_timeout = 0
        attempt_of_malformed_response = 0
        current_timeout = self.timeout

        while True:
            try:
                return await super()._agenerate(messages, stop, run_manager, timeout=current_timeout, **kwargs)
            except openai.RateLimitError as e:
                attempt += 1
                if self.max_retries_on_rate_limit < 0 or attempt <= self.max_retries_on_rate_limit:
                    delay = get_retry_delay(attempt, error=e, retry_delay=self.retry_delay)
                    logger.warning(
                        f"Rate limit error, retrying in {delay} seconds... "
                        f"attempt: {attempt} of "
                        f"{self.max_retries_on_rate_limit if self.max_retries_on_rate_limit >= 0 else 'infinite'}"
                    )
                    if run_manager:
                        await run_manager.on_retry(retry_state={"attempt": attempt, "delay": delay, "extracted_delay": extract_retry_seconds(e)})
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise e
            except openai.NotFoundError as e:
                attempt_on_not_found += 1
                if self.max_retries_on_not_found < 0 or attempt_on_not_found <= self.max_retries_on_not_found:
                    logger.warning(
                        f"Model not found error, possibly due to APIM, retrying in {self.retry_delay * 10} seconds... "
                        f"attempt: {attempt_on_not_found} of "
                        f"{self.max_retries_on_not_found if self.max_retries_on_not_found >= 0 else 'infinite'}"
                    )
                    if run_manager:
                        await run_manager.on_retry(
                            retry_state={"attempt": attempt_on_not_found, "delay": self.retry_delay * 10, "extracted_delay": self.retry_delay * 10}
                        )
                    await asyncio.sleep(self.retry_delay * 10)
                    continue
                else:
                    raise e
            except (openai.APIConnectionError, TimeoutError) as e:
                attempt_of_timeout += 1
                if self.max_retries_on_timeout < 0 or attempt_of_timeout <= self.max_retries_on_timeout:
                    delay = self.retry_delay * 2
                    logger.warning(
                        f"Timeout error, retrying in {delay} seconds... "
                        f"attempt: {attempt_of_timeout} of "
                        f"{self.max_retries_on_timeout if self.max_retries_on_timeout >= 0 else 'infinite'}; "
                        f"next timeout: {current_timeout * 2:.1f}s"
                    )
                    if run_manager:
                        run_manager.on_retry(error=e, retry_state={"attempt": attempt_of_timeout, "delay": delay, "extracted_delay": delay})
                    await asyncio.sleep(delay)
                    current_timeout *= 3  # 3*timeout for next retry
                    continue
                else:
                    raise e
            except (KeyError, TypeError, ValueError) as e:
                if not _is_malformed_chat_response_error(e):
                    raise e

                attempt_of_malformed_response += 1
                details = _extract_http_error_details(e)
                if self.max_retries_on_malformed_response < 0 or attempt_of_malformed_response <= self.max_retries_on_malformed_response:
                    delay = self.retry_delay * 2
                    logger.warning(
                        "Malformed chat-completions response, retrying in {} seconds... attempt: {} of {}. Details: {}",
                        delay,
                        attempt_of_malformed_response,
                        self.max_retries_on_malformed_response if self.max_retries_on_malformed_response >= 0 else "infinite",
                        details or {"error": str(e)},
                    )
                    if run_manager:
                        await run_manager.on_retry(
                            retry_state={"attempt": attempt_of_malformed_response, "delay": delay, "details": details or {"error": str(e)}}
                        )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        "Malformed chat-completions response after retries exhausted. Details: {}",
                        details or {"error": str(e)},
                    )
                    raise e


class ChatOpenAIO1(ChatOpenAIWithRetries):
    """ChatOpenAI wrapper that removes temperature parameter for o1"""

    @property
    def _default_params(self):
        params = super()._default_params
        params.pop("temperature", None)
        return params


class AzureOpenAIEmbeddingsWithRateLimitRetries(AzureOpenAIEmbeddings):
    max_retries: int = 5  # do NOT default to infinite for embeddings
    retry_delay: float = 1.0

    def embed_documents(self, texts: List[str], chunk_size: Union[int, None] = None) -> List[List[float]]:
        attempt = 0
        while attempt <= self.max_retries:
            try:
                return super().embed_documents(texts, chunk_size)
            except TRANSIENT_EXCEPTIONS as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                delay = get_retry_delay(attempt, error=e, retry_delay=self.retry_delay)
                logger.warning(f"Transient error embedding, retrying in {delay} seconds... ({attempt}/{self.max_retries})")
                time.sleep(delay)
            except AssertionError as e:
                # Treat as a connection/transport bug; retry a small number of times then fail loudly
                attempt += 1
                if attempt > min(self.max_retries, 2):
                    raise
                delay = get_retry_delay(attempt, error=e, retry_delay=self.retry_delay)
                logger.warning(f"SDK transport assertion during embeddings, retrying in {delay} seconds... ({attempt})")
                time.sleep(delay)


class OpenAICallbackHandlerWithTime(OpenAICallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.requests_time_in_seconds: List[float] = []
        self.rate_limit_delays: List[float] = []
        self._local = threading.local()

    def __repr__(self) -> str:
        avg_time = sum(self.requests_time_in_seconds) / len(self.requests_time_in_seconds) if self.requests_time_in_seconds else 0
        avg_delay = sum(self.rate_limit_delays) / len(self.rate_limit_delays) if self.rate_limit_delays else 0
        return (
            f"{super().__repr__()}\n"
            f"Total Request Time: {self.requests_time_in_seconds}\n"
            f"Average Request Time: {avg_time:.2f} seconds\n"
            f"Rate Limit Delays: {self.rate_limit_delays}\n"
            f"Average Rate Limit Delay: {avg_delay:.2f} seconds"
        )

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Record start time of LLM call."""
        self._local.start_time = time.time()
        super().on_llm_start(serialized, prompts, **kwargs)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # timing (your existing logic)
        if hasattr(self._local, "start_time"):
            with self._lock:
                self.requests_time_in_seconds.append(time.time() - self._local.start_time)
            delattr(self._local, "start_time")

        # let LangChain do its normal accounting first
        before_cached = getattr(self, "prompt_tokens_cached", 0)
        before_reasoning = getattr(self, "reasoning_tokens", 0)
        super().on_llm_end(response, **kwargs)

        # now try to read the detailed usage that includes caching/reasoning
        llm_out = getattr(response, "llm_output", None) or {}
        usage = llm_out.get("token_usage") or llm_out.get("usage") or {}

        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0

        # avoid double counting if super() already handled these in your version
        with self._lock:
            if getattr(self, "prompt_tokens_cached", 0) == before_cached:
                self.prompt_tokens_cached = before_cached + int(cached)
            if getattr(self, "reasoning_tokens", 0) == before_reasoning:
                self.reasoning_tokens = before_reasoning + int(reasoning)

    def __copy__(self) -> "OpenAICallbackHandlerWithTime":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "OpenAICallbackHandlerWithTime":
        """Return a deep copy of the callback handler."""
        return self

    def on_retry(self, retry_state: Dict[str, Any], **kwargs: Any) -> None:
        """Track rate limit delays."""
        if "delay" in retry_state:
            with self._lock:
                self.rate_limit_delays.append(retry_state["delay"])


@contextmanager
def get_openai_callback_with_time() -> Generator[OpenAICallbackHandlerWithTime, None, None]:
    """Get the OpenAI callback handler in a context manager.
    which conveniently exposes token and cost information.

    Returns:
        OpenAICallbackHandler: The OpenAI callback handler.

    Example:
        >>> with get_openai_callback() as cb:
        ...     # Use the OpenAI callback handler
    """
    cb = OpenAICallbackHandlerWithTime()
    openai_callback_var.set(cb)
    yield cb
    openai_callback_var.set(None)
