"""
Model-aware chat template system for preprocessing pipeline.

This module provides chat templates for different model architectures and
utilities for template detection, validation, and application.
"""

from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from enum import Enum
import re
import json
from .config_v2 import ChatTemplate


@dataclass
class TemplateValidationResult:
    """Result of template validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class ChatTemplateRegistry:
    """Registry for chat templates supporting various model architectures."""
    
    # Built-in chat templates
    TEMPLATES = {
        ChatTemplate.CHATML: {
            "template": "<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>",
            "description": "ChatML format used by OpenAI models and derivatives",
            "required_fields": ["query", "response"],
            "system_template": "<|im_start|>system\n{system}<|im_end|>\n",
        },
        ChatTemplate.LLAMA: {
            "template": "<s>[INST] {query} [/INST] {response} </s>",
            "description": "Llama chat format with instruction tokens",
            "required_fields": ["query", "response"],
            "system_template": "<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{query} [/INST] {response} </s>",
        },
        ChatTemplate.ALPACA: {
            "template": "### Instruction:\n{query}\n\n### Response:\n{response}",
            "description": "Alpaca instruction-following format",
            "required_fields": ["query", "response"],
            "system_template": "### System:\n{system}\n\n### Instruction:\n{query}\n\n### Response:\n{response}",
        },
        ChatTemplate.GEMMA: {
            "template": "<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>",
            "description": "Google Gemma chat format",
            "required_fields": ["query", "response"],
            "system_template": "<start_of_turn>system\n{system}<end_of_turn>\n<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>",
        },
        ChatTemplate.VICUNA: {
            "template": "USER: {query}\nASSISTANT: {response}",
            "description": "Vicuna conversation format",
            "required_fields": ["query", "response"],
            "system_template": "SYSTEM: {system}\nUSER: {query}\nASSISTANT: {response}",
        },
    }
    
    @classmethod
    def get_template(cls, template_type: ChatTemplate) -> Dict[str, Any]:
        """Get template configuration by type."""
        if template_type not in cls.TEMPLATES:
            raise ValueError(f"Unknown template type: {template_type}")
        return cls.TEMPLATES[template_type].copy()
    
    @classmethod
    def get_template_string(cls, template_type: ChatTemplate, use_system: bool = False) -> str:
        """Get the template string for formatting."""
        template_config = cls.get_template(template_type)
        if use_system and "system_template" in template_config:
            return template_config["system_template"]
        return template_config["template"]
    
    @classmethod
    def list_templates(cls) -> List[Dict[str, Any]]:
        """List all available templates with their descriptions."""
        return [
            {
                "type": template_type.value,
                "description": config["description"],
                "required_fields": config["required_fields"],
                "has_system": "system_template" in config,
            }
            for template_type, config in cls.TEMPLATES.items()
        ]
    
    @classmethod
    def validate_template(cls, template_type: ChatTemplate, sample_data: Dict[str, str]) -> TemplateValidationResult:
        """Validate that a template can be applied to sample data."""
        errors = []
        warnings = []
        
        try:
            template_config = cls.get_template(template_type)
            template_string = template_config["template"]
            required_fields = template_config["required_fields"]
            
            # Check required fields are present
            missing_fields = [field for field in required_fields if field not in sample_data]
            if missing_fields:
                errors.append(f"Missing required fields: {missing_fields}")
            
            # Try to format the template
            try:
                formatted = template_string.format(**sample_data)
                if not formatted.strip():
                    warnings.append("Template produces empty output")
            except KeyError as e:
                errors.append(f"Template formatting failed: missing key {e}")
            except Exception as e:
                errors.append(f"Template formatting error: {e}")
            
            # Check for potential issues
            if len(formatted) > 10000:
                warnings.append("Formatted template is very long (>10k chars)")
            
        except Exception as e:
            errors.append(f"Template validation error: {e}")
        
        return TemplateValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @classmethod
    def register_custom_template(
        cls, 
        name: str, 
        template: str, 
        description: str, 
        required_fields: List[str],
        system_template: Optional[str] = None
    ) -> None:
        """Register a custom template."""
        custom_type = ChatTemplate.CUSTOM
        cls.TEMPLATES[custom_type] = {
            "template": template,
            "description": description,
            "required_fields": required_fields,
            "name": name,
        }
        if system_template:
            cls.TEMPLATES[custom_type]["system_template"] = system_template


class ModelDetector:
    """Utility for detecting appropriate chat templates based on model names."""
    
    # Model name patterns mapped to templates
    MODEL_MAPPINGS = {
        # OpenAI and derivatives
        r"(gpt|chatgpt|openai)": ChatTemplate.CHATML,
        r"(dialogpt|dialo-gpt)": ChatTemplate.CHATML,
        
        # Meta Llama family
        r"(llama|llama2|llama-2|code-llama)": ChatTemplate.LLAMA,
        r"(meta-llama|facebook/llama)": ChatTemplate.LLAMA,
        
        # Google models
        r"(gemma|google/gemma)": ChatTemplate.GEMMA,
        
        # Alpaca family
        r"(alpaca|stanford-alpaca)": ChatTemplate.ALPACA,
        r"(tatsu-lab/alpaca)": ChatTemplate.ALPACA,
        
        # Vicuna family
        r"(vicuna|lmsys/vicuna)": ChatTemplate.VICUNA,
        
        # Other instruction-tuned models (default to Alpaca format)
        r"(instruct|instruction|chat)": ChatTemplate.ALPACA,
    }
    
    @classmethod
    def detect_template(cls, model_name: str) -> ChatTemplate:
        """Detect appropriate chat template for a model name."""
        model_lower = model_name.lower()
        
        for pattern, template in cls.MODEL_MAPPINGS.items():
            if re.search(pattern, model_lower):
                return template
        
        # Default fallback
        return ChatTemplate.CHATML
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """Get comprehensive model information including template and metadata."""
        template = cls.detect_template(model_name)
        template_config = ChatTemplateRegistry.get_template(template)
        
        return {
            "model_name": model_name,
            "detected_template": template.value,
            "template_description": template_config["description"],
            "required_fields": template_config["required_fields"],
            "supports_system": "system_template" in template_config,
            "confidence": cls._calculate_confidence(model_name, template),
        }
    
    @classmethod
    def _calculate_confidence(cls, model_name: str, template: ChatTemplate) -> float:
        """Calculate confidence score for template detection."""
        model_lower = model_name.lower()
        
        # High confidence for exact matches
        high_confidence_patterns = {
            ChatTemplate.LLAMA: [r"llama", r"meta-llama"],
            ChatTemplate.GEMMA: [r"gemma", r"google/gemma"],
            ChatTemplate.ALPACA: [r"alpaca", r"tatsu-lab/alpaca"],
            ChatTemplate.VICUNA: [r"vicuna", r"lmsys/vicuna"],
            ChatTemplate.CHATML: [r"gpt", r"openai", r"dialogpt"],
        }
        
        if template in high_confidence_patterns:
            for pattern in high_confidence_patterns[template]:
                if re.search(pattern, model_lower):
                    return 0.9
        
        # Medium confidence for partial matches
        medium_confidence_patterns = {
            ChatTemplate.ALPACA: [r"instruct", r"instruction", r"chat"],
            ChatTemplate.LLAMA: [r"code-llama"],
        }
        
        if template in medium_confidence_patterns:
            for pattern in medium_confidence_patterns[template]:
                if re.search(pattern, model_lower):
                    return 0.6
        
        # Low confidence (fallback)
        return 0.3


class TemplateFormatter:
    """Utility for applying templates to data with validation and error handling."""
    
    def __init__(self, template_type: ChatTemplate, use_system: bool = False):
        self.template_type = template_type
        self.use_system = use_system
        self.template_string = ChatTemplateRegistry.get_template_string(template_type, use_system)
        self.template_config = ChatTemplateRegistry.get_template(template_type)
    
    def format_sample(self, data: Dict[str, str]) -> str:
        """Format a single data sample using the template."""
        # Validate required fields
        required_fields = self.template_config["required_fields"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Apply template
        try:
            formatted = self.template_string.format(**data)
            return formatted
        except KeyError as e:
            raise ValueError(f"Template formatting failed: missing key {e}")
        except Exception as e:
            raise ValueError(f"Template formatting error: {e}")
    
    def format_batch(self, data_list: List[Dict[str, str]]) -> List[str]:
        """Format a batch of data samples."""
        results = []
        errors = []
        
        for i, data in enumerate(data_list):
            try:
                formatted = self.format_sample(data)
                results.append(formatted)
            except Exception as e:
                errors.append(f"Sample {i}: {e}")
                results.append(None)
        
        if errors:
            raise ValueError(f"Formatting errors: {'; '.join(errors)}")
        
        return results
    
    def validate_and_format(self, data: Dict[str, str]) -> tuple[bool, str, List[str]]:
        """Validate and format data, returning success status, result, and errors."""
        try:
            validation = ChatTemplateRegistry.validate_template(self.template_type, data)
            if not validation.is_valid:
                return False, "", validation.errors
            
            formatted = self.format_sample(data)
            return True, formatted, validation.warnings
        
        except Exception as e:
            return False, "", [str(e)]


# Utility functions
def create_template_formatter(model_name: str, use_system: bool = False) -> TemplateFormatter:
    """Create a template formatter for a specific model."""
    template_type = ModelDetector.detect_template(model_name)
    return TemplateFormatter(template_type, use_system)


def validate_model_template_compatibility(model_name: str, sample_data: Dict[str, str]) -> TemplateValidationResult:
    """Validate that a model's detected template works with sample data."""
    template_type = ModelDetector.detect_template(model_name)
    return ChatTemplateRegistry.validate_template(template_type, sample_data)


def get_template_examples() -> Dict[str, Dict[str, Any]]:
    """Get example formatted outputs for all templates."""
    sample_data = {
        "query": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "system": "You are a helpful assistant."
    }
    
    examples = {}
    for template_type in ChatTemplate:
        if template_type == ChatTemplate.CUSTOM:
            continue
            
        try:
            # Basic template
            formatter = TemplateFormatter(template_type, use_system=False)
            basic_example = formatter.format_sample(sample_data)
            
            # System template (if available)
            system_example = None
            try:
                system_formatter = TemplateFormatter(template_type, use_system=True)
                system_example = system_formatter.format_sample(sample_data)
            except:
                pass
            
            examples[template_type.value] = {
                "basic": basic_example,
                "with_system": system_example,
                "description": ChatTemplateRegistry.get_template(template_type)["description"]
            }
        except Exception as e:
            examples[template_type.value] = {
                "error": str(e)
            }
    
    return examples