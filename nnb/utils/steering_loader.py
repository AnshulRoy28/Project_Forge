"""Load and format steering documents for Gemini prompts."""

from pathlib import Path
from typing import Dict, List, Optional


class SteeringLoader:
    """Loads steering documents and formats them for Gemini context."""
    
    STEERING_DIR = Path(__file__).parent.parent.parent / ".kiro" / "steering"
    
    # Critical documents that should ALWAYS be included
    CRITICAL_DOCS = [
        "CRITICAL-RULES.md",  # HIGHEST PRIORITY
        "architecture.md",
        "coding-style.md",
    ]
    
    # Stage-specific document mappings
    STAGE_DOCS = {
        "conversation": ["CRITICAL-RULES.md", "architecture.md"],
        "scoping": ["CRITICAL-RULES.md", "architecture.md", "coding-style.md"],
        "data_requirements": ["CRITICAL-RULES.md", "architecture.md"],
        "data_validation": ["CRITICAL-RULES.md", "architecture.md"],
        "environment": ["CRITICAL-RULES.md", "architecture.md", "docker-patterns.md", "security.md"],
        "code_generation": [
            "CRITICAL-RULES.md",
            "architecture.md",
            "coding-style.md",
            "error-handling.md",
            "security.md",
        ],
        "training": ["CRITICAL-RULES.md", "architecture.md", "docker-patterns.md"],
        "inference": ["CRITICAL-RULES.md", "architecture.md", "coding-style.md", "security.md"],
    }
    
    @classmethod
    def load_document(cls, filename: str) -> str:
        """Load a single steering document."""
        filepath = cls.STEERING_DIR / filename
        
        if not filepath.exists():
            return f"# {filename} not found"
        
        content = filepath.read_text(encoding="utf-8")
        
        # Remove YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                content = parts[2].strip()
        
        return content
    
    @classmethod
    def get_critical_principles(cls) -> str:
        """Get the most critical principles that MUST be followed."""
        return """
CRITICAL PRINCIPLES (MUST FOLLOW):

1. THREE-LAYER ARCHITECTURE:
   - Orchestrator (CLI) - user interaction, state management
   - Docker Runtime - all code execution in isolated containers
   - Gemini Brain - staged prompts, role-specific per stage
   
2. IMMUTABILITY:
   - NEVER mutate objects
   - ALWAYS create new objects with changes
   - Use spread operators: {...obj, key: value}
   
3. STATE MACHINE:
   - Projects flow through 13 states: INIT → DONE
   - No skipping stages
   - Hard gates at critical points
   
4. ERROR HANDLING:
   - Handle ALL errors explicitly
   - Provide user-friendly messages
   - Log detailed context
   - Always suggest fixes
   
5. DOCKER ISOLATION:
   - One container per project
   - Data mounted read-only at /data
   - Workspace mounted read-write at /workspace
   - Host NEVER executes training code
   
6. CODE QUALITY:
   - Small files (200-400 lines, 800 max)
   - Small functions (<50 lines)
   - No deep nesting (>4 levels)
   - Comprehensive error handling
   - No hardcoded values
"""
    
    @classmethod
    def get_stage_context(cls, stage: str) -> str:
        """Get steering context for a specific stage."""
        docs_to_load = cls.STAGE_DOCS.get(stage, cls.CRITICAL_DOCS)
        
        context_parts = [
            "=" * 80,
            "STEERING DOCUMENTS - MUST FOLLOW THESE GUIDELINES",
            "=" * 80,
            "",
            cls.get_critical_principles(),
            "",
            "=" * 80,
            f"STAGE-SPECIFIC GUIDANCE: {stage.upper()}",
            "=" * 80,
            "",
        ]
        
        for doc in docs_to_load:
            content = cls.load_document(doc)
            context_parts.append(f"### From {doc}:")
            context_parts.append(content)
            context_parts.append("")
            context_parts.append("-" * 80)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    @classmethod
    def get_code_generation_rules(cls) -> str:
        """Get specific rules for code generation."""
        return """
CODE GENERATION RULES (MANDATORY):

1. IMMUTABILITY:
   ```python
   # WRONG
   def update(config, key, val):
       config[key] = val  # MUTATION!
       return config
   
   # CORRECT
   def update(config, key, val):
       return {**config, key: val}
   ```

2. ERROR HANDLING:
   ```python
   # ALWAYS include try-except with user-friendly messages
   try:
       result = operation()
   except SpecificError as e:
       logger.error(f"Operation failed: {e}", exc_info=True)
       print(f"❌ Operation failed: {e}")
       print(f"💡 Fix: {suggested_fix}")
       sys.exit(1)
   ```

3. FILE SIZE:
   - Target: 200-400 lines
   - Maximum: 800 lines
   - Extract utilities if larger

4. FUNCTION SIZE:
   - Maximum: 50 lines
   - Extract helper functions if larger

5. NO HARDCODED VALUES:
   - Use config files or constants
   - Never hardcode paths, URLs, or magic numbers

6. LOGGING:
   - Log all important operations
   - Use appropriate levels (DEBUG, INFO, WARNING, ERROR)
   - Include context in log messages

7. VALIDATION:
   - Validate all inputs at system boundaries
   - Use Pydantic for schema validation
   - Fail fast with clear messages

8. DOCKER AWARENESS:
   - All training code runs in containers
   - Data at /data (read-only)
   - Workspace at /workspace (read-write)
   - Never assume host filesystem access
"""
    
    @classmethod
    def get_docker_rules(cls) -> str:
        """Get Docker-specific rules."""
        return """
DOCKER RULES (MANDATORY):

1. CONTAINER LIFECYCLE:
   - One container per project
   - Built once in Stage 5, reused throughout
   - Rebuilding is explicit

2. VOLUME MOUNTS:
   - Data: read-only at /data
   - Workspace: read-write at /workspace
   - All generated files go to /workspace

3. DOCKERFILE GENERATION:
   - Select appropriate base image (PyTorch, TensorFlow, JAX)
   - Pin ALL dependency versions
   - Install ONLY required packages
   - No bloat, no unnecessary tools

4. HEALTH CHECKS:
   - Verify GPU/CPU availability
   - Test framework imports
   - Check data mount accessibility
   - Validate workspace permissions

5. ISOLATION:
   - Host NEVER runs training code
   - All execution in container
   - Files flow through volume mounts only
"""
    
    @classmethod
    def format_for_prompt(cls, stage: str, include_code_rules: bool = False) -> str:
        """Format steering context for inclusion in a Gemini prompt."""
        parts = [cls.get_stage_context(stage)]
        
        if include_code_rules:
            parts.append("")
            parts.append("=" * 80)
            parts.append("CODE GENERATION RULES")
            parts.append("=" * 80)
            parts.append(cls.get_code_generation_rules())
        
        if stage in ["environment", "code_generation", "training"]:
            parts.append("")
            parts.append("=" * 80)
            parts.append("DOCKER RULES")
            parts.append("=" * 80)
            parts.append(cls.get_docker_rules())
        
        return "\n".join(parts)
