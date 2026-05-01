# LLM Interface Module (M5b) Documentation

## Overview

The LLM Interface Module provides a standardized wrapper for local Large Language Model (LLM) operations using Ollama. It handles text generation, retry logic, and error handling, with support for reusing the Fusion Module's Mistral runner when available.

## Purpose

- Provide unified interface for LLM text generation
- Handle HTTP requests to Ollama API
- Implement retry logic with exponential backoff
- Parse and structure LLM responses
- Support multiple generation parameters
- Enable fallback to existing implementations

## Tools & Technologies

### Core Dependencies
- **Python 3.9+**: Primary programming language
- **Ollama**: Local LLM runtime
- **Requests**: HTTP client library
- **Mistral (via Ollama)**: Default LLM model

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB |
| CPU | 4 cores | 8+ cores |
| Ollama | Latest | Latest |
| Model | mistral (7B) | mistral (7B) or larger |

## Implementation Details

### Module Structure

```
llm_interface/
├── __init__.py
└── ollama_client.py             # Ollama API client
```

### Core Components

#### 1. OllamaResponse Data Structure

```python
@dataclass
class OllamaResponse:
    """Response from Ollama generation."""
    text: str                    # Generated text content
    model: str                   # Model name/identifier
    total_duration_ms: int       # Generation time in milliseconds
    prompt_eval_count: int       # Number of input tokens
    eval_count: int              # Number of generated tokens
```

#### 2. Exception Hierarchy

```python
class OllamaError(Exception):
    """Base exception for Ollama client errors."""

class OllamaConnectionError(OllamaError):
    """Network connection error."""

class OllamaTimeoutError(OllamaError):
    """Timeout during generation."""

class OllamaServerError(OllamaError):
    """Server error (5xx)."""

class OllamaClientError(OllamaError):
    """Client error (4xx)."""
```

#### 3. OllamaClient Class

**Main Client Class**: Handles all LLM operations

```python
class OllamaClient:
    """
    Ollama client wrapper with retry logic and error handling.

    Reuses Fusion Module's generate() function when available.
    """
```

**Default Configuration**:

```python
BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral"
DEFAULT_TIMEOUT = 120              # Read timeout (seconds)
DEFAULT_CONNECT_TIMEOUT = 5        # Connection timeout (seconds)
MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]           # Exponential backoff
```

**Generation Parameters**:

```python
DEFAULT_GENERATION_PARAMS = {
    "num_predict": 512,            # Maximum tokens to generate
    "temperature": 0.7,            # Sampling temperature
    "top_k": 40,                   # Top-k sampling
    "top_p": 0.9,                  # Top-p (nucleus) sampling
    "repeat_penalty": 1.1           # Penalize repetition
}

DEFAULT_JSON_OPTIONS = {
    "num_predict": 512,
    "temperature": 0.1,             # Lower for deterministic output
    "top_k": 40,
    "top_p": 0.9
}
```

#### 4. Initialization

```python
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
```

#### 5. Payload Building

```python
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
```

#### 6. HTTP Request Execution

```python
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
```

#### 7. Response Parsing

```python
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
```

#### 8. Generate Method with Retry Logic

```python
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

    # Use Fusion Module if available
    if FUSION_AVAILABLE:
        return self._generate_with_fusion_reuse(prompt, model_name)

    # Otherwise use new implementation
    return self._generate_with_retry(prompt, model_name)
```

**Retry Logic**:

```python
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
```

#### 9. Health Check

```python
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
```

## Configuration Parameters

### Client Configuration

```python
OllamaClient(
    base_url="http://localhost:11434",    # Ollama API endpoint
    model="mistral",                      # Model name
    timeout=120,                          # Request timeout (seconds)
    connect_timeout=5,                    # Connection timeout (seconds)
    max_retries=3                         # Maximum retry attempts
)
```

### Generation Parameters

```python
# Standard text generation
options = {
    "num_predict": 512,      # Max tokens to generate
    "temperature": 0.7,      # Sampling temperature
    "top_k": 40,             # Top-k sampling
    "top_p": 0.9,            # Nucleus sampling
    "repeat_penalty": 1.1    # Repetition penalty
}

# JSON/deterministic output
json_options = {
    "num_predict": 512,
    "temperature": 0.1,      # Lower for deterministic output
    "top_k": 40,
    "top_p": 0.9
}

# Creative output
creative_options = {
    "num_predict": 1024,     # Longer output
    "temperature": 0.9,      # Higher for creativity
    "top_k": 50,             # More diverse
    "top_p": 0.95,           # Wider sampling
    "repeat_penalty": 1.05   # Less penalty
}
```

## Usage Examples

### Basic Usage

```python
from llm_interface import OllamaClient

# Initialize client
client = OllamaClient(
    base_url="http://localhost:11434",
    model="mistral"
)

# Generate text
response = client.generate(
    prompt="Explain the concept of derivatives in calculus."
)

print(f"Generated: {response.text}")
print(f"Time: {response.total_duration_ms}ms")
print(f"Tokens: {response.eval_count}")
```

### With Custom Parameters

```python
# Using custom generation parameters
response = client.generate(
    prompt="Create a JSON object describing a graph.",
    options={
        "num_predict": 256,
        "temperature": 0.1,      # Lower for more deterministic
        "top_p": 0.9
    }
)
```

### Error Handling

```python
try:
    response = client.generate(prompt)
    print(f"Success: {response.text[:100]}...")
except OllamaConnectionError as e:
    print(f"Connection failed: {e}")
    # Check if Ollama is running
except OllamaTimeoutError as e:
    print(f"Request timed out: {e}")
    # Try with longer timeout or shorter prompt
except OllamaServerError as e:
    print(f"Server error: {e}")
    # Check Ollama logs
except OllamaClientError as e:
    print(f"Client error: {e}")
    # Fix prompt or parameters
```

### Health Check

```python
client = OllamaClient()

if client.check_connection():
    print("Ollama is ready")
    response = client.generate("Hello!")
else:
    print("Ollama is not available")
    # Start Ollama or check configuration
```

### Batch Processing

```python
def process_prompts(client, prompts):
    """Process multiple prompts with error handling."""
    results = []

    for i, prompt in enumerate(prompts):
        try:
            response = client.generate(prompt)
            results.append(response.text)
            print(f"Processed {i+1}/{len(prompts)}")

        except OllamaError as e:
            print(f"Error processing prompt {i+1}: {e}")
            results.append(None)

    return results

prompts = [
    "Explain concept A",
    "Explain concept B",
    "Explain concept C"
]

results = process_prompts(client, prompts)
```

## Performance Characteristics

### Processing Time

| Prompt Length | Tokens | Time | Notes |
|---------------|--------|------|-------|
| Short (50 words) | ~100 | 1-3s | Quick responses |
| Medium (200 words) | ~400 | 3-8s | Standard usage |
| Long (500 words) | ~1000 | 8-20s | Complex tasks |
| Very Long (1000+ words) | ~2000+ | 20-60s | Deep analysis |

### Throughput

| Model | Tokens/Second | Use Case |
|-------|---------------|----------|
| mistral (7B) | 50-100 | General purpose |
| llama2 (13B) | 30-60 | Higher quality |
| codellama (7B) | 40-80 | Code generation |

### Memory Usage

| Model | RAM | VRAM | Notes |
|-------|-----|------|-------|
| mistral (7B) | 8GB | 0GB | CPU-only |
| mistral (7B) | 4GB | 4GB | GPU-accelerated |
| llama2 (13B) | 16GB | 0GB | CPU-only |
| llama2 (13B) | 8GB | 8GB | GPU-accelerated |

## Troubleshooting

### Common Issues

**Issue**: "Cannot connect to Ollama"
- **Solution**: Check if Ollama is running
  ```bash
  # Start Ollama
  ollama serve

  # Check if running
  curl http://localhost:11434/api/tags
  ```

**Issue**: "Model not found"
- **Solution**: Pull the required model
  ```bash
  # Pull mistral
  ollama pull mistral

  # List available models
  ollama list
  ```

**Issue**: "Request timeout"
- **Solutions**:
  - Increase timeout parameter
  - Shorten prompt
  - Use smaller model
  - Check system resources

**Issue**: "Out of memory"
- **Solutions**:
  - Use smaller model
  - Reduce `num_predict` parameter
  - Close other applications
  - Use CPU-only mode

### Debugging

```python
# Enable detailed logging
import logging
import requests
logging.basicConfig(level=logging.DEBUG)

# Check connection
client = OllamaClient()
if not client.check_connection():
    print("Ollama is not running")
    # Start Ollama or check port

# Test with simple prompt
try:
    response = client.generate("Hello", options={"num_predict": 10})
    print(f"Test successful: {response.text}")
except Exception as e:
    print(f"Test failed: {e}")
    # Check error details

# Monitor performance
import time
start = time.time()
response = client.generate("Test prompt")
duration = time.time() - start
print(f"Generated {response.eval_count} tokens in {duration:.2f}s")
print(f"Speed: {response.eval_count/duration:.1f} tokens/s")
```

## Best Practices

### 1. Error Handling

```python
# Comprehensive error handling
def safe_generate(client, prompt, max_retries=3):
    """Generate with robust error handling."""
    for attempt in range(max_retries):
        try:
            return client.generate(prompt)
        except OllamaConnectionError:
            print("Connection lost, retrying...")
            time.sleep(2 ** attempt)
        except OllamaTimeoutError:
            print("Timeout, trying with shorter prompt...")
            prompt = prompt[:len(prompt)//2]
        except OllamaError as e:
            print(f"Fatal error: {e}")
            return None
    return None
```

### 2. Prompt Engineering

```python
# Clear, specific prompts
bad_prompt = "tell me about graphs"
good_prompt = """
Explain the key components of a line graph in 3 sentences:
1. What the axes represent
2. How to read the data
3. Common uses in education
"""

# Structured output for parsing
structured_prompt = """
Analyze this visual and return JSON:
{
  "content_type": "...",
  "description": "...",
  "elements": [...]
}

Visual data: {vlm_data}
"""
```

### 3. Parameter Tuning

```python
# Deterministic output (for JSON, code)
deterministic_params = {
    "temperature": 0.1,
    "top_k": 40,
    "top_p": 0.9,
    "num_predict": 512
}

# Creative output (for explanations, stories)
creative_params = {
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.95,
    "num_predict": 1024,
    "repeat_penalty": 1.0
}

# Concise output (for summaries)
concise_params = {
    "temperature": 0.5,
    "num_predict": 256,
    "repeat_penalty": 1.2
}
```

### 4. Resource Management

```python
# Monitor and manage resources
def generate_with_monitoring(client, prompt):
    """Generate while monitoring system resources."""
    import psutil

    # Check available memory before
    mem_before = psutil.virtual_memory().available / 1024**3
    print(f"Available RAM: {mem_before:.1f} GB")

    # Generate
    response = client.generate(prompt)

    # Check after
    mem_after = psutil.virtual_memory().available / 1024**3
    print(f"RAM used: {mem_before - mem_after:.1f} GB")

    return response
```

## Future Enhancements

1. **Streaming Support**: Process responses as they're generated
2. **Async Operations**: Non-blocking generation
3. **Batch Requests**: Process multiple prompts efficiently
4. **Caching**: Cache responses for repeated prompts
5. **Model Selection**: Automatic model selection based on task
6. **Progress Callbacks**: Real-time generation progress
7. **Custom Endpoints**: Support for alternative LLM backends
