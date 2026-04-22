---
inclusion: auto
description: Security best practices for the CLI Neural Network Builder including API key management, container isolation, and data protection.
---

# Security Guidelines

## Mandatory Security Checks

Before ANY commit:
- [ ] No hardcoded API keys (Gemini, cloud providers)
- [ ] All user inputs validated
- [ ] Path traversal prevention (file operations)
- [ ] Command injection prevention (Docker exec)
- [ ] Container isolation verified
- [ ] Data mount is read-only
- [ ] Secrets stored securely (environment variables, keyring)
- [ ] Error messages don't leak sensitive data

## API Key Management

### NEVER Hardcode Keys

```python
# WRONG: Hardcoded API key
GEMINI_API_KEY = "AIzaSyC..."

# CORRECT: Environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ConfigurationError("GEMINI_API_KEY not set")
```

### Secure Storage

```python
import keyring

# Store API key securely
keyring.set_password("nnb", "gemini_api_key", api_key)

# Retrieve API key
api_key = keyring.get_password("nnb", "gemini_api_key")
```

### Validation at Startup

```python
def validate_required_secrets():
    """Validate all required secrets are present."""
    required = ["GEMINI_API_KEY"]
    missing = [key for key in required if not os.getenv(key)]
    
    if missing:
        raise ConfigurationError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Set them with: export {missing[0]}=your-key-here"
        )
```

## Container Isolation

### Read-Only Data Mount

```python
# CRITICAL: Data must be read-only to prevent accidental modification
docker.run(
    volumes={
        str(data_path): {"bind": "/data", "mode": "ro"},  # READ-ONLY
        str(workspace_path): {"bind": "/workspace", "mode": "rw"}
    }
)
```

### Network Isolation

```python
# Disable network access during training (optional but recommended)
docker.run(
    network_mode="none",  # No network access
    volumes={...}
)
```

### Resource Limits

```python
# Prevent resource exhaustion
docker.run(
    mem_limit="8g",
    cpus=4,
    pids_limit=100,  # Prevent fork bombs
    volumes={...}
)
```

## Input Validation

### Path Traversal Prevention

```python
def validate_data_path(user_path: str) -> Path:
    """Validate user-provided path to prevent traversal attacks."""
    path = Path(user_path).resolve()
    
    # Ensure path is within allowed directories
    allowed_dirs = [Path.cwd(), Path.home() / "data"]
    
    if not any(path.is_relative_to(allowed) for allowed in allowed_dirs):
        raise SecurityError(f"Path {path} is outside allowed directories")
    
    if not path.exists():
        raise ValueError(f"Path {path} does not exist")
    
    return path
```

### Command Injection Prevention

```python
# WRONG: Vulnerable to command injection
def run_command(user_input):
    os.system(f"python train.py --config {user_input}")

# CORRECT: Use parameterized commands
def run_command(config_path):
    validated_path = validate_path(config_path)
    subprocess.run(
        ["python", "train.py", "--config", str(validated_path)],
        check=True
    )
```

### Schema Validation

```python
from pydantic import BaseModel, validator

class ProjectSpec(BaseModel):
    project_type: str
    framework: str
    num_classes: int
    
    @validator("project_type")
    def validate_project_type(cls, v):
        allowed = ["vision", "nlp", "tabular", "time-series"]
        if v not in allowed:
            raise ValueError(f"Invalid project_type: {v}")
        return v
    
    @validator("num_classes")
    def validate_num_classes(cls, v):
        if v < 2 or v > 1000:
            raise ValueError("num_classes must be between 2 and 1000")
        return v
```

## Gemini API Security

### Rate Limiting

```python
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=10, period=60)  # 10 calls per minute
def call_gemini_api(prompt):
    """Rate-limited Gemini API call."""
    return gemini_client.generate(prompt)
```

### Input Sanitization

```python
def sanitize_prompt(user_input: str) -> str:
    """Sanitize user input before sending to Gemini."""
    # Remove potential prompt injection attempts
    sanitized = user_input.replace("Ignore previous instructions", "")
    sanitized = sanitized.replace("System:", "")
    
    # Limit length
    max_length = 10000
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized
```

### Response Validation

```python
def validate_gemini_response(response: str) -> dict:
    """Validate Gemini response before execution."""
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        raise SecurityError("Invalid JSON response from Gemini")
    
    # Validate expected structure
    required_keys = ["status", "issues"]
    if not all(key in parsed for key in required_keys):
        raise SecurityError("Missing required keys in Gemini response")
    
    return parsed
```

## Data Protection

### Sensitive Data Detection

```python
import re

def detect_sensitive_data(text: str) -> list[str]:
    """Detect potential sensitive data in text."""
    patterns = {
        "api_key": r"(?i)(api[_-]?key|apikey)[\s:=]+['\"]?([a-zA-Z0-9_-]+)['\"]?",
        "password": r"(?i)(password|passwd|pwd)[\s:=]+['\"]?([^\s'\"]+)['\"]?",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    }
    
    findings = []
    for name, pattern in patterns.items():
        if re.search(pattern, text):
            findings.append(name)
    
    return findings
```

### Logging Sanitization

```python
def sanitize_log_message(message: str) -> str:
    """Remove sensitive data from log messages."""
    # Redact API keys
    message = re.sub(
        r"(api[_-]?key|apikey)[\s:=]+['\"]?([a-zA-Z0-9_-]+)['\"]?",
        r"\1=REDACTED",
        message,
        flags=re.IGNORECASE
    )
    
    # Redact passwords
    message = re.sub(
        r"(password|passwd|pwd)[\s:=]+['\"]?([^\s'\"]+)['\"]?",
        r"\1=REDACTED",
        message,
        flags=re.IGNORECASE
    )
    
    return message
```

## Error Handling

### Safe Error Messages

```python
# WRONG: Leaks sensitive information
try:
    gemini_client.generate(prompt)
except Exception as e:
    print(f"Error: {e}")  # Might contain API key or internal paths

# CORRECT: Generic user-facing message, detailed logging
try:
    gemini_client.generate(prompt)
except Exception as e:
    logger.error(f"Gemini API error: {e}", exc_info=True)
    print("❌ Failed to communicate with Gemini API. Check logs for details.")
    sys.exit(1)
```

## Security Response Protocol

If security issue found:
1. **STOP immediately** — Don't proceed with the operation
2. **Log the incident** — Record what happened
3. **Notify the user** — Clear, actionable message
4. **Fix CRITICAL issues** before continuing
5. **Rotate any exposed secrets** immediately
6. **Review entire codebase** for similar issues

## Dependency Security

### Pin Versions

```txt
# requirements.txt
google-genai==0.1.0  # Pinned version
docker==7.0.0
pydantic==2.5.0
```

### Regular Updates

```bash
# Check for security vulnerabilities
pip-audit

# Update dependencies
pip install --upgrade pip-audit
pip-audit --fix
```

### Minimal Dependencies

Only install what's absolutely necessary:

```txt
# WRONG: Too many dependencies
requests
beautifulsoup4
selenium
pandas
numpy
scipy
matplotlib

# CORRECT: Minimal set
google-genai
docker
pydantic
```

## Best Practices

### DO:
- ✅ Store secrets in environment variables or keyring
- ✅ Validate all user inputs
- ✅ Use read-only mounts for data
- ✅ Sanitize logs and error messages
- ✅ Rate limit API calls
- ✅ Pin dependency versions
- ✅ Run containers with resource limits

### DON'T:
- ❌ Hardcode API keys or secrets
- ❌ Trust user input without validation
- ❌ Allow write access to data directory
- ❌ Log sensitive information
- ❌ Execute arbitrary user commands
- ❌ Use unpinned dependencies
- ❌ Run containers with unlimited resources
