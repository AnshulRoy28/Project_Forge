"""
Unit tests for the chat template system.
"""

import pytest
from forge.core.templates import (
    ChatTemplateRegistry, ModelDetector, TemplateFormatter,
    create_template_formatter, validate_model_template_compatibility,
    get_template_examples, TemplateValidationResult
)
from forge.core.config_v2 import ChatTemplate


class TestChatTemplateRegistry:
    """Test ChatTemplateRegistry class."""
    
    def test_get_template(self):
        """Test getting template configuration."""
        template_config = ChatTemplateRegistry.get_template(ChatTemplate.CHATML)
        
        assert "template" in template_config
        assert "description" in template_config
        assert "required_fields" in template_config
        assert template_config["required_fields"] == ["query", "response"]
    
    def test_get_template_string(self):
        """Test getting template string for formatting."""
        # Basic template
        template_str = ChatTemplateRegistry.get_template_string(ChatTemplate.CHATML)
        assert "{query}" in template_str
        assert "{response}" in template_str
        
        # System template
        system_template_str = ChatTemplateRegistry.get_template_string(ChatTemplate.CHATML, use_system=True)
        assert "{system}" in system_template_str
        assert "{query}" in system_template_str
        assert "{response}" in system_template_str
    
    def test_list_templates(self):
        """Test listing all available templates."""
        templates = ChatTemplateRegistry.list_templates()
        
        assert len(templates) > 0
        for template in templates:
            assert "type" in template
            assert "description" in template
            assert "required_fields" in template
            assert "has_system" in template
    
    def test_validate_template_valid(self):
        """Test template validation with valid data."""
        sample_data = {
            "query": "What is the capital of France?",
            "response": "The capital of France is Paris."
        }
        
        result = ChatTemplateRegistry.validate_template(ChatTemplate.CHATML, sample_data)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_template_missing_fields(self):
        """Test template validation with missing required fields."""
        sample_data = {
            "query": "What is the capital of France?"
            # Missing "response" field
        }
        
        result = ChatTemplateRegistry.validate_template(ChatTemplate.CHATML, sample_data)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("Missing required fields" in error for error in result.errors)
    
    def test_validate_template_empty_output(self):
        """Test template validation with data that produces empty output."""
        sample_data = {
            "query": "",
            "response": ""
        }
        
        result = ChatTemplateRegistry.validate_template(ChatTemplate.CHATML, sample_data)
        
        # Should be valid but with warnings
        assert result.is_valid
        assert len(result.warnings) > 0
    
    def test_register_custom_template(self):
        """Test registering a custom template."""
        ChatTemplateRegistry.register_custom_template(
            name="test_custom",
            template="Q: {question}\nA: {answer}",
            description="Test custom template",
            required_fields=["question", "answer"]
        )
        
        custom_config = ChatTemplateRegistry.get_template(ChatTemplate.CUSTOM)
        assert custom_config["name"] == "test_custom"
        assert custom_config["template"] == "Q: {question}\nA: {answer}"
        assert custom_config["required_fields"] == ["question", "answer"]


class TestModelDetector:
    """Test ModelDetector class."""
    
    def test_detect_template_llama(self):
        """Test detecting Llama template."""
        test_cases = [
            "meta-llama/Llama-2-7b-chat",
            "llama-7b-instruct",
            "code-llama-13b",
            "facebook/llama-65b"
        ]
        
        for model_name in test_cases:
            template = ModelDetector.detect_template(model_name)
            assert template == ChatTemplate.LLAMA, f"Failed for {model_name}"
    
    def test_detect_template_gemma(self):
        """Test detecting Gemma template."""
        test_cases = [
            "google/gemma-7b-it",
            "gemma-2b-instruct",
            "google/gemma-1.1-7b-it"
        ]
        
        for model_name in test_cases:
            template = ModelDetector.detect_template(model_name)
            assert template == ChatTemplate.GEMMA, f"Failed for {model_name}"
    
    def test_detect_template_chatml(self):
        """Test detecting ChatML template."""
        test_cases = [
            "microsoft/DialoGPT-medium",
            "openai/gpt-3.5-turbo",
            "gpt-4-turbo",
            "chatgpt-instruct"
        ]
        
        for model_name in test_cases:
            template = ModelDetector.detect_template(model_name)
            assert template == ChatTemplate.CHATML, f"Failed for {model_name}"
    
    def test_detect_template_alpaca(self):
        """Test detecting Alpaca template."""
        test_cases = [
            "tatsu-lab/alpaca-7b",
            "stanford-alpaca",
            "alpaca-lora-7b",
            "some-instruct-model",
            "instruction-following-model"
        ]
        
        for model_name in test_cases:
            template = ModelDetector.detect_template(model_name)
            assert template == ChatTemplate.ALPACA, f"Failed for {model_name}"
    
    def test_detect_template_vicuna(self):
        """Test detecting Vicuna template."""
        test_cases = [
            "lmsys/vicuna-7b-v1.5",
            "vicuna-13b-delta",
            "lmsys/vicuna-33b"
        ]
        
        for model_name in test_cases:
            template = ModelDetector.detect_template(model_name)
            assert template == ChatTemplate.VICUNA, f"Failed for {model_name}"
    
    def test_detect_template_fallback(self):
        """Test fallback to ChatML for unknown models."""
        unknown_models = [
            "unknown/model",
            "custom-model-v1",
            "experimental/test-model"
        ]
        
        for model_name in unknown_models:
            template = ModelDetector.detect_template(model_name)
            assert template == ChatTemplate.CHATML, f"Failed fallback for {model_name}"
    
    def test_get_model_info(self):
        """Test getting comprehensive model information."""
        model_info = ModelDetector.get_model_info("meta-llama/Llama-2-7b-chat")
        
        assert model_info["model_name"] == "meta-llama/Llama-2-7b-chat"
        assert model_info["detected_template"] == "llama"
        assert "template_description" in model_info
        assert "required_fields" in model_info
        assert "supports_system" in model_info
        assert "confidence" in model_info
        assert 0 <= model_info["confidence"] <= 1
    
    def test_confidence_calculation(self):
        """Test confidence calculation for template detection."""
        # High confidence cases
        high_conf_info = ModelDetector.get_model_info("meta-llama/Llama-2-7b")
        assert high_conf_info["confidence"] >= 0.8
        
        # Medium confidence cases
        medium_conf_info = ModelDetector.get_model_info("some-instruct-model")
        assert 0.5 <= medium_conf_info["confidence"] < 0.8
        
        # Low confidence cases (fallback)
        low_conf_info = ModelDetector.get_model_info("unknown-model")
        assert low_conf_info["confidence"] < 0.5


class TestTemplateFormatter:
    """Test TemplateFormatter class."""
    
    def test_format_sample_chatml(self):
        """Test formatting a sample with ChatML template."""
        formatter = TemplateFormatter(ChatTemplate.CHATML)
        
        data = {
            "query": "What is 2+2?",
            "response": "2+2 equals 4."
        }
        
        result = formatter.format_sample(data)
        
        assert "<|im_start|>user" in result
        assert "What is 2+2?" in result
        assert "<|im_start|>assistant" in result
        assert "2+2 equals 4." in result
        assert "<|im_end|>" in result
    
    def test_format_sample_llama(self):
        """Test formatting a sample with Llama template."""
        formatter = TemplateFormatter(ChatTemplate.LLAMA)
        
        data = {
            "query": "Hello, how are you?",
            "response": "I'm doing well, thank you!"
        }
        
        result = formatter.format_sample(data)
        
        assert "<s>[INST]" in result
        assert "Hello, how are you?" in result
        assert "[/INST]" in result
        assert "I'm doing well, thank you!" in result
        assert "</s>" in result
    
    def test_format_sample_with_system(self):
        """Test formatting a sample with system prompt."""
        formatter = TemplateFormatter(ChatTemplate.LLAMA, use_system=True)
        
        data = {
            "query": "What is AI?",
            "response": "AI stands for Artificial Intelligence.",
            "system": "You are a helpful assistant."
        }
        
        result = formatter.format_sample(data)
        
        assert "<<SYS>>" in result
        assert "You are a helpful assistant." in result
        assert "<</SYS>>" in result
        assert "What is AI?" in result
        assert "AI stands for Artificial Intelligence." in result
    
    def test_format_sample_missing_fields(self):
        """Test formatting with missing required fields."""
        formatter = TemplateFormatter(ChatTemplate.CHATML)
        
        data = {
            "query": "What is 2+2?"
            # Missing "response" field
        }
        
        with pytest.raises(ValueError, match="Missing required fields"):
            formatter.format_sample(data)
    
    def test_format_batch(self):
        """Test formatting a batch of samples."""
        formatter = TemplateFormatter(ChatTemplate.ALPACA)
        
        data_list = [
            {"query": "What is 1+1?", "response": "1+1 equals 2."},
            {"query": "What is 2+2?", "response": "2+2 equals 4."},
            {"query": "What is 3+3?", "response": "3+3 equals 6."}
        ]
        
        results = formatter.format_batch(data_list)
        
        assert len(results) == 3
        for result in results:
            assert "### Instruction:" in result
            assert "### Response:" in result
    
    def test_format_batch_with_errors(self):
        """Test formatting a batch with some invalid samples."""
        formatter = TemplateFormatter(ChatTemplate.ALPACA)
        
        data_list = [
            {"query": "What is 1+1?", "response": "1+1 equals 2."},
            {"query": "What is 2+2?"},  # Missing response
            {"query": "What is 3+3?", "response": "3+3 equals 6."}
        ]
        
        with pytest.raises(ValueError, match="Formatting errors"):
            formatter.format_batch(data_list)
    
    def test_validate_and_format(self):
        """Test validation and formatting combined."""
        formatter = TemplateFormatter(ChatTemplate.GEMMA)
        
        # Valid data
        valid_data = {
            "query": "Hello!",
            "response": "Hi there!"
        }
        
        success, result, warnings = formatter.validate_and_format(valid_data)
        
        assert success
        assert "<start_of_turn>user" in result
        assert "Hello!" in result
        assert "<start_of_turn>model" in result
        assert "Hi there!" in result
        
        # Invalid data
        invalid_data = {
            "query": "Hello!"
            # Missing response
        }
        
        success, result, errors = formatter.validate_and_format(invalid_data)
        
        assert not success
        assert result == ""
        assert len(errors) > 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_template_formatter(self):
        """Test creating template formatter from model name."""
        # Test Llama model
        formatter = create_template_formatter("meta-llama/Llama-2-7b")
        assert formatter.template_type == ChatTemplate.LLAMA
        
        # Test with system prompt
        system_formatter = create_template_formatter("meta-llama/Llama-2-7b", use_system=True)
        assert system_formatter.use_system == True
    
    def test_validate_model_template_compatibility(self):
        """Test validating model-template compatibility."""
        sample_data = {
            "query": "Test query",
            "response": "Test response"
        }
        
        # Should work for any model with valid data
        result = validate_model_template_compatibility("microsoft/DialoGPT-medium", sample_data)
        assert result.is_valid
        
        # Should fail with missing data
        invalid_data = {"query": "Test query"}
        result = validate_model_template_compatibility("microsoft/DialoGPT-medium", invalid_data)
        assert not result.is_valid
    
    def test_get_template_examples(self):
        """Test getting template examples."""
        examples = get_template_examples()
        
        # Should have examples for all non-custom templates
        expected_templates = [t.value for t in ChatTemplate if t != ChatTemplate.CUSTOM]
        
        for template_type in expected_templates:
            assert template_type in examples
            assert "basic" in examples[template_type]
            assert "description" in examples[template_type]
            
            # Check that basic example is a string
            if "error" not in examples[template_type]:
                assert isinstance(examples[template_type]["basic"], str)


class TestTemplateProperties:
    """Property-based tests for template system."""
    
    def test_template_formatting_consistency(self):
        """Property: Template formatting should be consistent and reversible."""
        test_data = {
            "query": "What is the meaning of life?",
            "response": "The meaning of life is 42.",
            "system": "You are a helpful assistant."
        }
        
        # Test all templates
        for template_type in [ChatTemplate.CHATML, ChatTemplate.LLAMA, ChatTemplate.ALPACA, ChatTemplate.GEMMA, ChatTemplate.VICUNA]:
            formatter = TemplateFormatter(template_type)
            
            # Should format without errors
            result = formatter.format_sample(test_data)
            assert isinstance(result, str)
            assert len(result) > 0
            
            # Should contain the original data
            assert test_data["query"] in result
            assert test_data["response"] in result
    
    def test_model_detection_consistency(self):
        """Property: Model detection should be consistent and deterministic."""
        test_models = [
            "meta-llama/Llama-2-7b-chat",
            "google/gemma-7b-it",
            "microsoft/DialoGPT-medium",
            "tatsu-lab/alpaca-7b",
            "lmsys/vicuna-13b"
        ]
        
        for model_name in test_models:
            # Detection should be consistent across multiple calls
            template1 = ModelDetector.detect_template(model_name)
            template2 = ModelDetector.detect_template(model_name)
            assert template1 == template2
            
            # Model info should be consistent
            info1 = ModelDetector.get_model_info(model_name)
            info2 = ModelDetector.get_model_info(model_name)
            assert info1["detected_template"] == info2["detected_template"]
            assert info1["confidence"] == info2["confidence"]
    
    def test_validation_consistency(self):
        """Property: Validation should be consistent and meaningful."""
        valid_data = {
            "query": "Test question",
            "response": "Test answer"
        }
        
        invalid_data = {
            "query": "Test question"
            # Missing response
        }
        
        for template_type in [ChatTemplate.CHATML, ChatTemplate.LLAMA, ChatTemplate.ALPACA]:
            # Valid data should always validate
            result = ChatTemplateRegistry.validate_template(template_type, valid_data)
            assert result.is_valid
            
            # Invalid data should always fail
            result = ChatTemplateRegistry.validate_template(template_type, invalid_data)
            assert not result.is_valid
            assert len(result.errors) > 0