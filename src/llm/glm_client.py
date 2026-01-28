"""GLM-4.7 LangChain compatible client for Zhipu AI."""
from typing import Any, List, Optional, Iterator
import httpx
import logging
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    AIMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field, SecretStr
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

logger = logging.getLogger(__name__)

# Maximum characters for embedding input (Zhipu API has ~8192 token limit)
MAX_EMBEDDING_CHARS = 6000


class RateLimitError(Exception):
    """Rate limit error that should be retried."""
    pass


def _is_retryable_error(exception: BaseException) -> bool:
    """Check if the exception is retryable (not 400 Bad Request or ValueError)."""
    # Retry rate limit errors
    if isinstance(exception, RateLimitError):
        return True
    if isinstance(exception, ValueError) and "rate limit" in str(exception).lower():
        return True
    # Don't retry other ValueErrors (typically validation or API format errors)
    if isinstance(exception, ValueError):
        return False
    if isinstance(exception, httpx.HTTPStatusError):
        # Don't retry client errors (4xx) except 429 (rate limit)
        if 400 <= exception.response.status_code < 500:
            return exception.response.status_code == 429
    return True


class GLMChat(BaseChatModel):
    """LangChain compatible chat model for Zhipu AI GLM-4.7."""

    api_key: SecretStr = Field(..., description="Zhipu AI API key")
    base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4/",
        description="API base URL"
    )
    model: str = Field(default="glm-4.7", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=16384)  # Large for reasoning models
    timeout: int = Field(default=600)  # 10 minutes for complex analysis

    @property
    def _llm_type(self) -> str:
        return "glm-4.7"

    @property
    def _identifying_params(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to API format."""
        converted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                converted.append({"role": "assistant", "content": msg.content})
            else:
                converted.append({"role": "user", "content": str(msg.content)})
        return converted

    @retry(
        stop=stop_after_attempt(5),  # More attempts for rate limit
        wait=wait_exponential(multiplier=2, min=5, max=60),  # Longer wait for rate limit (up to 60s)
        retry=retry_if_exception(_is_retryable_error)
    )
    def _call_api(self, messages: List[dict]) -> dict:
        """Make API call with retry logic."""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                
                # Log error details before raising
                if response.status_code != 200:
                    error_detail = response.text
                    logger.error(f"GLM API error {response.status_code}: {error_detail}")
                    if response.status_code == 401:
                        raise ValueError("Authentication failed - check your GLM_API_KEY")
                    elif response.status_code == 400:
                        raise ValueError(f"Bad request to GLM API: {error_detail}")
                    elif response.status_code == 429:
                        # Parse error message
                        try:
                            error_data = response.json()
                            error_msg = error_data.get("error", {}).get("message", "Rate limit exceeded")
                        except:
                            error_msg = "Rate limit exceeded"
                        
                        logger.warning(f"Rate limit hit: {error_msg}. Will retry with exponential backoff.")
                        raise RateLimitError(f"Rate limit exceeded: {error_msg}")
                
                response.raise_for_status()
                return response.json()
        except RateLimitError:
            # Re-raise rate limit errors (will be retried by tenacity)
            raise
        except httpx.ConnectError as e:
            # Network connection error
            error_msg = "GLM API connection failed. Please check your internet connection and API endpoint."
            logger.error(f"{error_msg} Original error: {e}")
            raise ValueError(error_msg) from e
        except httpx.HTTPStatusError as e:
            # Re-raise with more context
            error_msg = f"GLM API HTTP error {e.response.status_code}"
            if e.response.text:
                try:
                    error_data = e.response.json()
                    if "error" in error_data:
                        error_msg += f": {error_data['error']}"
                except:
                    error_msg += f": {e.response.text[:200]}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion."""
        converted_messages = self._convert_messages(messages)
        response = self._call_api(converted_messages)

        msg_data = response["choices"][0]["message"]
        content = msg_data.get("content", "")
        
        # GLM-4.7 and similar models may return reasoning in reasoning_content
        # If content is empty but reasoning_content exists, use that
        if not content and "reasoning_content" in msg_data:
            reasoning = msg_data["reasoning_content"]
            # Extract the final answer from reasoning if possible
            content = reasoning
            logger.debug("Using reasoning_content as response")
        
        message = AIMessage(content=content)

        generation = ChatGeneration(
            message=message,
            generation_info={
                "finish_reason": response["choices"][0].get("finish_reason"),
                "usage": response.get("usage", {}),
            },
        )

        return ChatResult(
            generations=[generation],
            llm_output={"model": self.model, "usage": response.get("usage", {})},
        )


class GLMEmbeddings(Embeddings):
    """LangChain compatible embeddings for Zhipu AI."""

    def __init__(
        self,
        api_key: SecretStr,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4/",
        model: str = "embedding-2",
        timeout: int = 60,
        **kwargs
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable_error)
    )
    def _embed_single(self, text: str, url: str, headers: dict) -> List[float]:
        """Embed a single text with retry logic."""
        # Truncate text if too long to avoid API errors
        if len(text) > MAX_EMBEDDING_CHARS:
            logger.warning(f"Truncating text from {len(text)} to {MAX_EMBEDDING_CHARS} chars")
            text = text[:MAX_EMBEDDING_CHARS]

        # Skip empty or whitespace-only text
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding, using placeholder")
            text = "[empty]"

        payload = {
            "model": self.model,
            "input": text,  # Single string for Zhipu API
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            if response.status_code == 401:
                raise ValueError("Authentication failed - check your API key")
            if response.status_code == 400:
                # Log the error details for debugging
                error_detail = response.text
                logger.error(f"API 400 error: {error_detail}")
                logger.error(f"Text length: {len(text)}, first 100 chars: {text[:100]}")
                raise ValueError(f"Bad request to embedding API: {error_detail}")
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Make embedding API calls - process one at a time for Zhipu API."""
        url = f"{self.base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        all_embeddings = []
        for text in texts:
            embedding = self._embed_single(text, url, headers)
            all_embeddings.append(embedding)

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        # Process in batches to avoid API limits
        batch_size = 16
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._call_api([text])[0]


class OpenAIEmbeddings(Embeddings):
    """LangChain compatible embeddings for OpenAI."""

    def __init__(
        self,
        api_key: SecretStr,
        model: str = "text-embedding-3-small",
        timeout: int = 60,
        **kwargs
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = "https://api.openai.com/v1"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable_error)
    )
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Make embedding API call to OpenAI."""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        # Preprocess texts
        processed_texts = []
        for text in texts:
            # Truncate if too long (OpenAI has 8191 token limit)
            if len(text) > MAX_EMBEDDING_CHARS:
                logger.warning(f"Truncating text from {len(text)} to {MAX_EMBEDDING_CHARS} chars")
                text = text[:MAX_EMBEDDING_CHARS]
            # Handle empty text
            if not text or not text.strip():
                text = "[empty]"
            processed_texts.append(text)

        payload = {
            "model": self.model,
            "input": processed_texts,
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            if response.status_code == 401:
                raise ValueError("Authentication failed - check your OpenAI API key")
            if response.status_code == 400:
                error_detail = response.text
                logger.error(f"OpenAI API 400 error: {error_detail}")
                raise ValueError(f"Bad request to OpenAI embedding API: {error_detail}")
            response.raise_for_status()
            data = response.json()

            # Sort by index to maintain order
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in embeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        # OpenAI supports batch requests, process in chunks of 2048
        batch_size = 100  # Safe batch size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._call_api([text])[0]
