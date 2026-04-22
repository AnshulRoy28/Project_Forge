---
inclusion: auto
description: CRITICAL rules that MUST be followed at all times - highest priority steering document
---

# CRITICAL RULES - MUST FOLLOW

> **This document contains the most critical rules that MUST be followed at all times.**
> **These rules take precedence over all other considerations.**

## 🚨 ARCHITECTURE (NON-NEGOTIABLE)

### Three-Layer Separation

```
┌─────────────────────────────────────────┐
│         ORCHESTRATOR (CLI)              │
│  - User interaction                     │
│  - State management                     │
│  - Gemini communication                 │
│  - NEVER executes training code         │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼──────┐  ┌─────▼────────┐
│   GEMINI    │  │    DOCKER    │
│   BRAIN     │  │   RUNTIME    │
│             │  │              │
│ - Staged    │  │ - Isolated   │
│   prompts   │  │   execution  │
│ - Role per  │  │ - All code   │
│   stage     │  │   runs here  │
└─────────────┘  └──────────────┘
```

**RULES:**
1. ❌ **NEVER** execute training code on host
2. ✅ **ALWAYS** execute in Docker container
3. ✅ **ALWAYS** use volume mounts for file transfer
4. ✅ **ALWAYS** keep layers decoupled

### State Machine (MUST RESPECT)

```
INIT → SCOPING → SPEC_CONFIRMED → DATA_REQUIRED → DATA_VALIDATED
     → ENV_BUILDING → ENV_READY → MOCK_RUNNING → MOCK_PASSED
     → TRAINING → TRAINING_COMPLETE → INFERENCE_READY → DONE
```

**RULES:**
1. ❌ **NEVER** skip states
2. ❌ **NEVER** proceed without passing hard gates
3. ✅ **ALWAYS** check current state before operations
4. ✅ **ALWAYS** save state after transitions

### Hard Gates (BLOCKING)

These MUST pass before proceeding:

1. **Spec Confirmation** - User must explicitly confirm
2. **Data Validation** - Data must pass format/structure checks
3. **Container Health** - Container must be healthy
4. **Mock Run** - Mock run must succeed
5. **Training Complete** - Training must finish successfully

## 🚨 IMMUTABILITY (CRITICAL)

### The Golden Rule

**NEVER MUTATE. ALWAYS CREATE NEW.**

```python
# ❌ WRONG - MUTATION
def update_config(config, key, value):
    config[key] = value  # MUTATION!
    return config

# ✅ CORRECT - IMMUTABILITY
def update_config(config, key, value):
    return {**config, key: value}
```

### Why This Matters

- Prevents hidden side effects
- Makes debugging possible
- Enables safe concurrency
- Required for Docker isolation

### Common Violations to Avoid

```python
# ❌ WRONG
list.append(item)
dict[key] = value
obj.property = value

# ✅ CORRECT
new_list = [*list, item]
new_dict = {**dict, key: value}
new_obj = replace(obj, property=value)
```

## 🚨 DOCKER ISOLATION (MANDATORY)

### Container Rules

1. **One container per project** - Not per command
2. **Built once** - In Stage 5, reused throughout
3. **Data read-only** - Mounted at `/data`
4. **Workspace read-write** - Mounted at `/workspace`
5. **Host isolation** - Host NEVER runs training code

### Volume Mount Pattern

```
Host                          Container
─────────────────────────────────────────
.nnb/workspace/      ←→      /workspace/     (read-write)
./user-data/         ←→      /data/          (read-only)
```

**RULES:**
1. ❌ **NEVER** modify data in container
2. ❌ **NEVER** run training on host
3. ✅ **ALWAYS** use volume mounts
4. ✅ **ALWAYS** generate files to /workspace

## 🚨 ERROR HANDLING (REQUIRED)

### Every Error Must Be Handled

```python
# ❌ WRONG - Silent failure
try:
    operation()
except:
    pass

# ✅ CORRECT - Comprehensive handling
try:
    operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    console.print(f"[red]❌ Operation failed: {e}[/red]")
    console.print(f"[yellow]💡 Fix: {suggested_fix}[/yellow]")
    sys.exit(1)
```

### Error Message Requirements

Every error message MUST include:
1. ❌ What failed (clear description)
2. 💡 How to fix it (actionable steps)
3. 📋 Context (what was being attempted)
4. 🆔 Error ID (for debugging)

## 🚨 CODE QUALITY (ENFORCED)

### File Size Limits

- **Target**: 200-400 lines
- **Maximum**: 800 lines
- **Action**: Extract utilities if larger

### Function Size Limits

- **Maximum**: 50 lines
- **Action**: Extract helpers if larger

### Nesting Limits

- **Maximum**: 4 levels
- **Action**: Extract functions if deeper

### Validation Requirements

```python
# ✅ ALWAYS validate at system boundaries
def process_user_input(data: str) -> ProcessedData:
    # Validate format
    if not data:
        raise ValueError("Input cannot be empty")
    
    # Validate schema
    validated = InputSchema.parse(data)
    
    # Process
    return process(validated)
```

## 🚨 SECURITY (NON-NEGOTIABLE)

### API Keys

1. ❌ **NEVER** hardcode API keys
2. ❌ **NEVER** log API keys
3. ✅ **ALWAYS** use system keyring
4. ✅ **ALWAYS** support environment variables

### Data Protection

1. ❌ **NEVER** modify user data
2. ❌ **NEVER** send data to third parties
3. ✅ **ALWAYS** mount data read-only
4. ✅ **ALWAYS** validate inputs

### Container Security

1. ✅ **ALWAYS** use minimal base images
2. ✅ **ALWAYS** pin dependency versions
3. ✅ **ALWAYS** set resource limits
4. ✅ **ALWAYS** run health checks

## 🚨 GEMINI PROMPTS (STRUCTURED)

### Prompt Structure

Every Gemini prompt MUST include:

```python
prompt = f"""
{steering_context}  # ← CRITICAL: Include steering rules

================================================================================
YOUR TASK: {task_name}
================================================================================

{task_description}

REMEMBER:
- {critical_rule_1}
- {critical_rule_2}
- {critical_rule_3}

Output format:
{expected_format}
"""
```

### Context Management

1. ✅ **ALWAYS** include steering context
2. ✅ **ALWAYS** specify output format
3. ✅ **ALWAYS** include critical reminders
4. ❌ **NEVER** assume Gemini remembers previous context

### Output Validation

```python
# ✅ ALWAYS validate Gemini output
response = gemini.generate(prompt)

# Validate format
if expected_format == "json":
    validated = json.loads(response)
    
# Validate schema
result = ResponseSchema.parse(validated)

# Validate business rules
if not result.is_valid():
    raise ValidationError("Invalid response")
```

## 🚨 TESTING (MANDATORY)

### Coverage Requirements

- **Minimum**: 80% code coverage
- **Required**: Unit + Integration + E2E tests
- **TDD**: Write tests BEFORE implementation

### Test Structure

```python
def test_feature():
    # Arrange
    setup_test_data()
    
    # Act
    result = feature_under_test()
    
    # Assert
    assert result.is_correct()
    
    # Cleanup
    cleanup_test_data()
```

## 🚨 LOGGING (REQUIRED)

### Log Levels

```python
logger.debug("Detailed diagnostic info")      # Development only
logger.info("Normal operation")                # Important events
logger.warning("Something unexpected")         # Potential issues
logger.error("Operation failed", exc_info=True)  # Errors with traceback
logger.critical("System failure")              # Severe errors
```

### What to Log

✅ **DO LOG:**
- State transitions
- API calls (without keys)
- File operations
- Errors with context
- Performance metrics

❌ **DON'T LOG:**
- API keys or secrets
- User passwords
- Sensitive data
- Full file contents

## 🚨 CHECKLIST BEFORE COMMIT

Before committing ANY code, verify:

- [ ] Follows three-layer architecture
- [ ] Respects state machine
- [ ] Uses immutable patterns
- [ ] Handles all errors
- [ ] Includes user-friendly messages
- [ ] Validates all inputs
- [ ] No hardcoded values
- [ ] No API keys in code
- [ ] Files under 800 lines
- [ ] Functions under 50 lines
- [ ] Tests written and passing
- [ ] Logging added
- [ ] Documentation updated

## 🚨 WHEN IN DOUBT

If you're unsure about anything:

1. **Check the steering documents** in `.kiro/steering/`
2. **Follow the architecture** - three layers, state machine
3. **Be immutable** - never mutate
4. **Handle errors** - always provide fixes
5. **Ask for clarification** - don't guess

---

**Remember: These rules exist for a reason. Following them ensures:**
- ✅ Secure, isolated execution
- ✅ Predictable, debuggable behavior
- ✅ Maintainable, testable code
- ✅ Excellent user experience

**Violating these rules will result in:**
- ❌ Security vulnerabilities
- ❌ Unpredictable failures
- ❌ Unmaintainable code
- ❌ Poor user experience

---

**Last Updated**: 2026-04-23  
**Priority**: HIGHEST  
**Status**: MANDATORY
