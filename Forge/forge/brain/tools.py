"""
Function calling tools for Gemini in Forge.
"""

from typing import Any

# Tool definitions for Gemini function calling
FORGE_TOOLS = [
    {
        "name": "generate_training_config",
        "description": "Generate a forge.yaml training configuration based on user goals and hardware",
        "parameters": {
            "type": "object",
            "properties": {
                "base_model": {
                    "type": "string",
                    "description": "The base model to fine-tune (e.g., 'unsloth/gemma-2b')",
                },
                "quantization": {
                    "type": "string",
                    "enum": ["4bit", "8bit", "none"],
                    "description": "Quantization level based on VRAM",
                },
                "lora_rank": {
                    "type": "integer",
                    "description": "LoRA rank (higher = more capacity but more VRAM)",
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Training batch size",
                },
                "learning_rate": {
                    "type": "number",
                    "description": "Learning rate",
                },
                "num_epochs": {
                    "type": "integer",
                    "description": "Number of training epochs",
                },
            },
            "required": ["base_model", "quantization", "lora_rank", "batch_size", "learning_rate"],
        },
    },
    {
        "name": "analyze_dataset",
        "description": "Analyze a dataset for quality, bias, and formatting issues",
        "parameters": {
            "type": "object",
            "properties": {
                "quality_score": {
                    "type": "number",
                    "description": "Quality score from 0-100",
                },
                "issues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of identified issues",
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of recommendations",
                },
                "estimated_tokens": {
                    "type": "integer",
                    "description": "Estimated total tokens in dataset",
                },
            },
            "required": ["quality_score", "issues", "recommendations"],
        },
    },
    {
        "name": "suggest_hyperparameters",
        "description": "Suggest optimal hyperparameters based on hardware and dataset",
        "parameters": {
            "type": "object",
            "properties": {
                "learning_rate": {
                    "type": "number",
                    "description": "Recommended learning rate",
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Recommended batch size",
                },
                "gradient_accumulation_steps": {
                    "type": "integer",
                    "description": "Recommended gradient accumulation steps",
                },
                "max_seq_length": {
                    "type": "integer",
                    "description": "Recommended maximum sequence length",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation for recommendations",
                },
            },
            "required": ["learning_rate", "batch_size", "max_seq_length", "reasoning"],
        },
    },
    {
        "name": "diagnose_training_issue",
        "description": "Diagnose a training issue and suggest fixes",
        "parameters": {
            "type": "object",
            "properties": {
                "issue_type": {
                    "type": "string",
                    "enum": ["oom", "nan_loss", "slow_convergence", "overfitting", "other"],
                    "description": "Type of issue detected",
                },
                "root_cause": {
                    "type": "string",
                    "description": "Root cause analysis",
                },
                "fix_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Steps to fix the issue",
                },
                "config_changes": {
                    "type": "object",
                    "description": "Suggested config changes as key-value pairs",
                },
            },
            "required": ["issue_type", "root_cause", "fix_steps"],
        },
    },
]


def get_tools_schema() -> list[dict[str, Any]]:
    """Get the tools schema for Gemini function calling."""
    return FORGE_TOOLS


def parse_function_call(response: dict) -> tuple[str, dict]:
    """
    Parse a function call from Gemini response.
    
    Returns (function_name, arguments).
    """
    if "function_call" not in response:
        raise ValueError("No function call in response")
    
    fc = response["function_call"]
    return fc["name"], fc.get("args", {})
