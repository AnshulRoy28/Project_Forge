---
inclusion: auto
description: Comprehensive error handling patterns for the CLI Neural Network Builder including recovery strategies and user communication.
---

# Error Handling

## Core Principles

1. **Fail Fast** — Detect errors as early as possible
2. **Clear Messages** — User-friendly error messages with actionable fixes
3. **Detailed Logging** — Technical details in logs, not terminal output
4. **Graceful Degradation** — Recover when possible, fail cleanly when not
5. **State Preservation** — Always save state before failing

## Error Categories

### User Errors (Recoverable)

Errors caused by user input or configuration:

```python
class UserError(Exception):
    """Base class for user-facing errors."""
    def __init__(self, message: str, fix: str):
        self.message = message
        self.fix = fix
        super().__init__(message)

class InvalidDataPathError(UserError):
    """User provided invalid data path."""
    pass

class MissingAPIKeyError(UserError):
    """API key not configured."""
    pass
```

**Handling:**
```python
try:
    validate_data_path(user_path)
except InvalidDataPathError as e:
    print(f"❌ {e.message}")
    print(f"💡 Fix: {e.fix}")
    sys.exit(1)
```

### System Errors (Potentially Recoverable)

Errors from external systems (Docker, Gemini API):

```python
class SystemError(Exception):
    """Base class for system errors."""
    def __init__(self, message: str, retry_possible: bool = False):
        self.message = message
        self.retry_possible = retry_possible
        super().__init__(message)

class DockerBuildError(SystemError):
    """Docker container build failed."""
    pass

class GeminiAPIError(SystemError):
    """Gemini API call failed."""
    pass
```

**Handling with Retry:**
```python
@retry(max_attempts=3, backoff=2.0)
def call_gemini_api(prompt):
    try:
        return gemini_client.generate(prompt)
    except GeminiAPIError as e:
        if e.retry_possible:
            logger.warning(f"Gemini API error, retrying: {e}")
            raise  # Trigger retry
        else:
            logger.error(f"Gemini API error, not retrying: {e}")
            print(f"❌ Failed to communicate with Gemini API")
            sys.exit(1)
```

### Internal Errors (Not Recoverable)

Programming errors or unexpected states:

```python
class InternalError(Exception):
    """Base class for internal errors."""
    pass

class InvalidStateTransitionError(InternalError):
    """Attempted invalid state transition."""
    pass

class CodeGenerationError(InternalError):
    """Generated code is invalid."""
    pass
```

**Handling:**
```python
try:
    transition_state(current, next_state)
except InvalidStateTransitionError as e:
    logger.error(f"Internal error: {e}", exc_info=True)
    print(f"❌ Internal error occurred. Please report this bug.")
    print(f"📋 Error ID: {error_id}")
    sys.exit(1)
```

## Stage-Specific Error Handling

### Stage 1-2: Conversation Errors

```python
def handle_conversation_error(e: Exception):
    """Handle errors during user conversation."""
    if isinstance(e, KeyboardInterrupt):
        print("\n⚠️  Conversation interrupted")
        print("💡 Resume with: nnb resume <project-id>")
        save_conversation_state()
        sys.exit(0)
    elif isinstance(e, GeminiAPIError):
        print("❌ Failed to communicate with Gemini")
        print("💡 Check your API key and internet connection")
        sys.exit(1)
    else:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print("❌ Unexpected error occurred")
        sys.exit(1)
```

### Stage 4: Data Validation Errors

```python
def handle_validation_error(e: DataValidationError):
    """Handle data validation errors."""
    print(f"❌ Data validation failed: {e.message}")
    print(f"\n📋 Issues found:")
    
    for issue in e.issues:
        severity_icon = "🔴" if issue.severity == "error" else "🟡"
        print(f"  {severity_icon} {issue.message}")
        print(f"     Fix: {issue.fix}")
    
    print(f"\n💡 Fix the issues above and run: nnb data validate --path {e.data_path}")
    sys.exit(1)
```

### Stage 5: Docker Build Errors

```python
def handle_docker_build_error(e: DockerBuildError):
    """Handle Docker build errors."""
    logger.error(f"Docker build failed: {e}", exc_info=True)
    
    print("❌ Failed to build Docker container")
    print(f"\n📋 Build log:")
    print(e.build_log)
    
    # Common fixes
    print(f"\n💡 Common fixes:")
    print("  1. Check Docker is running: docker ps")
    print("  2. Check disk space: df -h")
    print("  3. Try rebuilding: nnb env rebuild")
    
    sys.exit(1)
```

### Stage 6: Mock Run Errors

```python
def handle_mock_run_error(e: MockRunError, retry_count: int, max_retries: int):
    """Handle mock run errors with auto-retry."""
    logger.error(f"Mock run failed (attempt {retry_count}/{max_retries}): {e}")
    
    if retry_count < max_retries:
        print(f"⚠️  Mock run failed (attempt {retry_count}/{max_retries})")
        print(f"🔧 Analyzing error and generating fix...")
        
        # Ask Gemini to fix
        fix = gemini_brain.debug_mock_run(e.traceback)
        apply_fix(fix)
        
        print(f"✓ Fix applied, retrying...")
        return True  # Retry
    else:
        print(f"❌ Mock run failed after {max_retries} attempts")
        print(f"\n📋 Final error:")
        print(e.traceback)
        print(f"\n💡 Manual intervention required")
        print(f"   1. Check generated code in .nnb/workspace/")
        print(f"   2. Fix the error manually")
        print(f"   3. Run: nnb mock-run")
        sys.exit(1)
```

### Stage 7: Training Errors

```python
def handle_training_error(e: TrainingError):
    """Handle training errors."""
    logger.error(f"Training failed: {e}", exc_info=True)
    
    print("❌ Training failed")
    print(f"\n📋 Error:")
    print(e.message)
    
    # Ask Gemini for diagnosis
    print(f"\n🔍 Analyzing error...")
    diagnosis = gemini_brain.diagnose_training_error(e.logs)
    
    print(f"\n💡 Diagnosis:")
    print(diagnosis.explanation)
    print(f"\n🔧 Suggested fixes:")
    for i, fix in enumerate(diagnosis.fixes, 1):
        print(f"  {i}. {fix}")
    
    print(f"\n📁 Training logs saved to: .nnb/logs/training.log")
    sys.exit(1)
```

## Retry Patterns

### Exponential Backoff

```python
def retry_with_backoff(func, max_attempts=3, base_delay=1.0):
    """Retry function with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
            time.sleep(delay)
```

### Conditional Retry

```python
def should_retry(error: Exception) -> bool:
    """Determine if error is retryable."""
    retryable_errors = (
        GeminiAPIError,
        DockerNetworkError,
        TemporaryFileError,
    )
    
    non_retryable_errors = (
        InvalidAPIKeyError,
        DataValidationError,
        UserCancelledError,
    )
    
    if isinstance(error, non_retryable_errors):
        return False
    
    if isinstance(error, retryable_errors):
        return True
    
    # Default: don't retry unknown errors
    return False
```

## State Preservation

### Save State Before Failing

```python
def safe_operation(project, operation):
    """Perform operation with state preservation."""
    # Save current state
    checkpoint = project.save_checkpoint()
    
    try:
        result = operation()
        project.commit_checkpoint()
        return result
    except Exception as e:
        # Restore previous state
        project.restore_checkpoint(checkpoint)
        logger.error(f"Operation failed, state restored: {e}")
        raise
```

### Conversation History Preservation

```python
def save_conversation_on_error(conversation):
    """Save conversation history even on error."""
    try:
        # ... conversation logic ...
    except Exception as e:
        # Always save conversation before exiting
        conversation.save()
        raise
    finally:
        # Ensure conversation is saved
        if not conversation.is_saved():
            conversation.save()
```

## Logging Strategy

### Structured Logging

```python
import logging
import json

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def log_error(self, error, context=None):
        """Log error with structured context."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "traceback": traceback.format_exc()
        }
        
        self.logger.error(json.dumps(log_entry))
```

### Log Levels

```python
# DEBUG: Detailed diagnostic information
logger.debug(f"Validating data at {data_path}")

# INFO: General informational messages
logger.info(f"Starting Stage 6: Code Generation")

# WARNING: Something unexpected but recoverable
logger.warning(f"Mock run failed, retrying (attempt 2/3)")

# ERROR: Error that prevents operation
logger.error(f"Data validation failed: {e}", exc_info=True)

# CRITICAL: Severe error requiring immediate attention
logger.critical(f"Container health check failed: {e}")
```

## User Communication

### Error Message Format

```python
def format_error_message(error: Exception) -> str:
    """Format error for user display."""
    return f"""
❌ {error.title}

{error.description}

💡 How to fix:
{error.fix_instructions}

📁 Logs: {error.log_path}
🆔 Error ID: {error.error_id}
"""
```

### Progress Indication

```python
from rich.progress import Progress

def long_running_operation():
    """Show progress for long operations."""
    with Progress() as progress:
        task = progress.add_task("Building container...", total=100)
        
        try:
            for step in build_steps:
                step.execute()
                progress.update(task, advance=20)
        except Exception as e:
            progress.stop()
            handle_error(e)
```

## Best Practices

### DO:
- ✅ Catch specific exceptions, not generic `Exception`
- ✅ Provide actionable error messages
- ✅ Log detailed context for debugging
- ✅ Save state before risky operations
- ✅ Retry transient errors automatically
- ✅ Show progress for long operations
- ✅ Include error IDs for bug reports

### DON'T:
- ❌ Silently swallow errors
- ❌ Show technical tracebacks to users
- ❌ Retry non-retryable errors
- ❌ Lose state on error
- ❌ Use generic error messages
- ❌ Log sensitive information
- ❌ Exit without cleanup
