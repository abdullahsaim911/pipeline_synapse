"""
LLM Interface Module (M5b)

Local LLM client wrapper for Ollama API - handles generation requests, retries, and error handling.
Reuses Fusion Module's mistral_runner.py as base implementation.
"""

import os
import time
import requests
from dataclasses import dataclass
from typing import Optional

# Reuse existing function from Fusion Module
# Import path needs to be verified at runtime
FUSION_AVAILABLE = False
base_generate = None

try:
    import importlib.util
    import sys

    # Build path to Fusion Module's mistral_runner
    fusion_path = "Fusion-Module--Synapse-/llm/mistral_runner.py"
    full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), fusion_path)

    if os.path.exists(full_path):
        spec = importlib.util.spec_from_file_location("mistral_runner", full_path)
        if spec and spec.loader:
            mistral_module = importlib.util.module_from_spec(spec)
            sys.modules["mistral_runner"] = mistral_module
            spec.loader.exec_module(mistral_module)

            if hasattr(mistral_module, 'generate'):
                base_generate = mistral_module.generate
                FUSION_AVAILABLE = True
                print("[LLM Interface] Fusion Module's mistral_runner.py loaded successfully")

except Exception as e:
    print(f"[LLM Interface] Fusion Module's mistral_runner.py not available: {e}")
    print("[LLM Interface] Will use new Ollama client implementation")


@dataclass
class OllamaResponse:
    """Response from Ollama generation."""
    text: str
    model: str
    total_duration_ms: int
    prompt_eval_count: int
    eval_count: int


class OllamaError(Exception):
    """Base exception for Ollama client errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Network connection error."""
    pass


class OllamaTimeoutError(OllamaError):
    """Timeout during generation."""
    pass


class OllamaServerError(OllamaError):
    """Server error (5xx)."""
    pass


class OllamaClientError(OllamaError):
    """Client error (4xx)."""
    pass


class OllamaClient:
    """
    Ollama client wrapper with retry logic and error handling.

    Reuses Fusion Module's generate() function when available.
    """

    # Default configuration
    BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "mistral"
    DEFAULT_TIMEOUT = 120  # seconds (read timeout)
    DEFAULT_CONNECT_TIMEOUT = 5  # seconds (connect timeout)
    MAX_RETRIES = 3
    RETRY_DELAYS = [1, 2, 4]
    keep_alive=0  # seconds (exponential backoff)

    # Default generation options
    DEFAULT_GENERATION_PARAMS = {
        "num_predict": 512,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repeat_penalty": 1.1
    }

    # For structured JSON output
    DEFAULT_JSON_OPTIONS = {
        "num_predict": 512,
        "temperature": 0.1,  # Lower temperature for deterministic output
        "top_k": 40,
        "top_p": 0.9
    }

    def __init__(
        self,
        base_url: str = BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
        connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
        max_retries: int = MAX_RETRIES
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama API base URL
            model: Model name (default: mistral)
            timeout: Request timeout in seconds
            connect_timeout: Connection timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.max_retries = max_retries

        print(f"[LLM Interface] Initialized: {base_url}, model={model}, timeout={timeout}s")

    def _build_payload(
        self,
        prompt: str,
        stream: bool = False,
        options: Optional[dict] = None
    ) -> dict:
        """
        Build JSON payload for Ollama API.

        Args:
            prompt: Text prompt to send
            stream: Whether to stream response
            options: Optional generation parameters

        Returns:
            Complete JSON payload dictionary
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream
        }

        # Merge with default options
        default_options = self.DEFAULT_GENERATION_PARAMS.copy()
        if options:
            default_options.update(options)

        payload["options"] = default_options

        # Validate required fields
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        return payload

    def _make_request(self, payload: dict):
        """
        Execute HTTP POST request to Ollama API.

        Args:
            payload: Complete JSON payload

        Returns:
            Parsed JSON response

        Raises:
            OllamaConnectionError: Connection or network errors
            OllamaTimeoutError: Request timeout
            OllamaServerError: Server errors (5xx)
            OllamaClientError: Client errors (4xx)
        """
        url = f"{self.base_url}/api/generate"

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=(self.connect_timeout, self.timeout)
            )

            # Check for HTTP errors
            if 400 <= response.status_code < 500:
                raise OllamaClientError(
                    f"Client error {response.status_code}: {response.text}"
                )

            response.raise_for_status()

            return response.json()

        except requests.exceptions.Timeout as e:
            raise OllamaTimeoutError(f"Request timeout: {e}")

        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(f"Connection failed: {e}")

        except requests.exceptions.HTTPError as e:
            # Already handled client errors above
            if 400 <= e.response.status_code < 500:
                raise
            else:
                raise OllamaServerError(f"Server error {e.response.status_code}")

        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(f"Request failed: {e}")

    def _parse_response(self, response_json: dict) -> OllamaResponse:
        """
        Parse Ollama response into structured format.

        Args:
            response_json: Raw JSON response from Ollama

        Returns:
            OllamaResponse with structured data
        """
        text = response_json.get("response", "")
        model = response_json.get("model", self.model)
        done = response_json.get("done", True)

        # Extract timing info if available
        total_duration_ms = 0
        if "total_duration" in response_json:
            total_duration_ms = int(response_json["total_duration"] / 1_000_000)

        prompt_eval_count = 0
        if "prompt_eval_count" in response_json:
            prompt_eval_count = int(response_json["prompt_eval_count"])

        eval_count = 0
        if "eval_count" in response_json:
            eval_count = int(response_json["eval_count"])

        return OllamaResponse(
            text=text,
            model=model,
            total_duration_ms=total_duration_ms,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count
        )

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        stream: bool = False,
        options: Optional[dict] = None
    ) -> OllamaResponse:
        """
        Generate text using Ollama.

        Args:
            prompt: Text prompt to send
            model: Model name (uses default if not specified)
            stream: Whether to stream response (default: False)
            options: Optional generation parameters

        Returns:
            OllamaResponse with generated text and metadata
        """
        model_name = model or self.model

        if FUSION_AVAILABLE:
            return self._generate_with_fusion_reuse(prompt, model_name)
        else:
            return self._generate_with_retry(prompt, model_name)

    def _generate_with_fusion_reuse(
        self,
        prompt: str,
        model_name: str
    ) -> OllamaResponse:
        """
        Generate using Fusion Module's existing generate() function.
        Adds structured response wrapper and timing.
        """
        start_time = time.perf_counter()

        # Call Fusion's existing function
        text = base_generate(prompt)

        # Calculate timing
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        # Return structured response
        return OllamaResponse(
            text=text,
            model=model_name,
            total_duration_ms=duration_ms,
            prompt_eval_count=len(prompt.split()),  # Approximate
            eval_count=len(text.split())
        )

    def _generate_with_retry(
        self,
        prompt: str,
        model_name: str
    ) -> OllamaResponse:
        """
        Generate with exponential backoff retry logic.

        Args:
            prompt: Text prompt to send
            model_name: Model name

        Returns:
            OllamaResponse with generated text and metadata
        """
        payload = self._build_payload(prompt)

        for attempt in range(self.max_retries):
            try:
                start_time = time.perf_counter()
                response_json = self._make_request(payload)
                duration_ms = int((time.perf_counter() - start_time) * 1000)

                return self._parse_response(response_json)

            except (OllamaTimeoutError, OllamaConnectionError, OllamaServerError) as e:
                # Retry logic for retryable errors
                if attempt < self.max_retries - 1:
                    wait_time = self.RETRY_DELAYS[attempt]
                    print(f"[LLM Interface] Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                    print(f"[LLM Interface] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[LLM Interface] Failed after {self.max_retries} attempts: {e}")
                    raise

            except (OllamaClientError) as e:
                # Client errors don't retry
                print(f"[LLM Interface] Client error (no retry): {e}")
                raise

    def check_connection(self) -> bool:
        """
        Health check for Ollama service.

        Returns:
            True if Ollama is responding, False otherwise
        """
        url = f"{self.base_url}/api/tags"

        try:
            response = requests.get(url, timeout=self.connect_timeout)
            is_connected = response.status_code == 200

            if is_connected:
                print("[LLM Interface] Ollama is ready!")
            else:
                print(f"[LLM Interface] Ollama responded with status {response.status_code}")

            return is_connected

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"[LLM Interface] Cannot connect to Ollama: {e}")
            return False
