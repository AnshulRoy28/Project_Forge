"""
Unit tests for the enhanced configuration system.
"""

import pytest
import tempfile
from pathlib import Path
from forge.core.config_v2 import (
    ForgeConfig, ModelConfig, HardwareConfig, PreprocessingConfig,
    ChatTemplate, GPUArchitecture, create_default_config,
    validate_config_file, _detect_model_info, _calculate_batch_size
)


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_valid_model_config(self):
        """Test creating a valid model configuration."""
        config = ModelConfig(
            name="microsoft/DialoGPT-medium",
            architecture="gpt2",
            chat_template=ChatTemplate.CHATML,
            max_length=2048
        )
        
        assert config.name == "microsoft/DialoGPT-medium"
        assert config.architecture == "gpt2"
        assert config.chat_template == ChatTemplate.CHATML
        assert config.max_length == 2048
        assert config.special_tokens == {}
    
    def test_model_config_string_template(self):
        """Test that string templates are converted to enum."""
        config = ModelConfig(
            name="test-model",
            architecture="transformer",
            chat_template="llama",  # String instead of enum
            max_length=1024
        )
        
        assert config.chat_template == ChatTemplate.LLAMA
    
    def test_invalid_max_length(self):
        """Test that invalid max_length raises error."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            ModelConfig(
                name="test-model",
                architecture="transformer",
                chat_template=ChatTemplate.CHATML,
                max_length=0
            )


class TestHardwareConfig:
    """Test HardwareConfig class."""
    
    def test_valid_hardware_config(self):
        """Test creating a valid hardware configuration."""
        config = HardwareConfig(
            gpu_arch=GPUArchitecture.ADA,
            vram_gb=16.0,
            compute_capability=8.9,
            recommended_batch_size=24  # Updated for aggressive batching
        )
        
        assert config.gpu_arch == GPUArchitecture.ADA
        assert config.vram_gb == 16.0
        assert config.compute_capability == 8.9
        assert config.recommended_batch_size == 24
        assert config.use_gpu_preprocessing == True
        assert config.max_memory_usage == 0.8
    
    def test_hardware_config_string_arch(self):
        """Test that string architecture is converted to enum."""
        config = HardwareConfig(
            gpu_arch="blackwell",  # String instead of enum
            vram_gb=24.0,
            compute_capability=12.0,
            recommended_batch_size=8
        )
        
        assert config.gpu_arch == GPUArchitecture.BLACKWELL
    
    def test_invalid_vram(self):
        """Test that invalid VRAM raises error."""
        with pytest.raises(ValueError, match="vram_gb must be positive"):
            HardwareConfig(
                gpu_arch=GPUArchitecture.ADA,
                vram_gb=-1.0,
                compute_capability=8.9,
                recommended_batch_size=24  # Updated for consistency
            )
    
    def test_invalid_memory_usage(self):
        """Test that invalid memory usage raises error."""
        with pytest.raises(ValueError, match="max_memory_usage must be between 0 and 1"):
            HardwareConfig(
                gpu_arch=GPUArchitecture.ADA,
                vram_gb=16.0,
                compute_capability=8.9,
                recommended_batch_size=24,  # Updated for consistency
                max_memory_usage=1.5
            )


class TestPreprocessingConfig:
    """Test PreprocessingConfig class."""
    
    def test_valid_preprocessing_config(self):
        """Test creating a valid preprocessing configuration."""
        config = PreprocessingConfig(
            train_split=0.8,
            validation_split=0.2,
            test_split=0.0,
            chunk_size=500
        )
        
        assert config.train_split == 0.8
        assert config.validation_split == 0.2
        assert config.test_split == 0.0
        assert config.chunk_size == 500
        assert config.quality_checks == True
    
    def test_invalid_split_ratios(self):
        """Test that invalid split ratios raise error."""
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            PreprocessingConfig(
                train_split=0.8,
                validation_split=0.3,  # Sum > 1.0
                test_split=0.0
            )
    
    def test_invalid_chunk_size(self):
        """Test that invalid chunk size raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            PreprocessingConfig(chunk_size=0)
    
    def test_invalid_text_lengths(self):
        """Test that invalid text lengths raise error."""
        with pytest.raises(ValueError, match="min_text_length must be non-negative"):
            PreprocessingConfig(min_text_length=-1)
        
        with pytest.raises(ValueError, match="max_text_length must be greater than min_text_length"):
            PreprocessingConfig(min_text_length=100, max_text_length=50)


class TestForgeConfig:
    """Test ForgeConfig class."""
    
    def test_valid_forge_config(self):
        """Test creating a valid complete configuration."""
        model_config = ModelConfig(
            name="microsoft/DialoGPT-medium",
            architecture="gpt2",
            chat_template=ChatTemplate.CHATML
        )
        
        hardware_config = HardwareConfig(
            gpu_arch=GPUArchitecture.ADA,
            vram_gb=16.0,
            compute_capability=8.9,
            recommended_batch_size=24  # Updated for consistency
        )
        
        preprocessing_config = PreprocessingConfig()
        
        config = ForgeConfig(
            model=model_config,
            hardware=hardware_config,
            preprocessing=preprocessing_config
        )
        
        assert config.model == model_config
        assert config.hardware == hardware_config
        assert config.preprocessing == preprocessing_config
        assert config.version == "2.0"
    
    def test_config_serialization(self):
        """Test configuration serialization to/from dictionary."""
        model_config = ModelConfig(
            name="test-model",
            architecture="transformer",
            chat_template=ChatTemplate.LLAMA
        )
        
        hardware_config = HardwareConfig(
            gpu_arch=GPUArchitecture.AMPERE,
            vram_gb=12.0,
            compute_capability=8.6,
            recommended_batch_size=2
        )
        
        preprocessing_config = PreprocessingConfig(
            train_split=0.9,
            validation_split=0.1
        )
        
        original_config = ForgeConfig(
            model=model_config,
            hardware=hardware_config,
            preprocessing=preprocessing_config
        )
        
        # Serialize to dict
        config_dict = original_config.to_dict()
        
        # Deserialize from dict
        restored_config = ForgeConfig.from_dict(config_dict)
        
        # Check that all values are preserved
        assert restored_config.model.name == original_config.model.name
        assert restored_config.model.chat_template == original_config.model.chat_template
        assert restored_config.hardware.gpu_arch == original_config.hardware.gpu_arch
        assert restored_config.hardware.vram_gb == original_config.hardware.vram_gb
        assert restored_config.preprocessing.train_split == original_config.preprocessing.train_split
    
    def test_config_file_operations(self):
        """Test saving and loading configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            # Create test configuration
            original_config = create_default_config(
                model_name="microsoft/DialoGPT-medium",
                gpu_arch=GPUArchitecture.ADA,
                vram_gb=16.0,
                compute_capability=8.9
            )
            
            # Save configuration
            original_config.save(config_path)
            assert config_path.exists()
            
            # Load configuration
            loaded_config = ForgeConfig.load(config_path)
            
            # Verify loaded configuration matches original
            assert loaded_config.model.name == original_config.model.name
            assert loaded_config.hardware.vram_gb == original_config.hardware.vram_gb
            assert loaded_config.preprocessing.train_split == original_config.preprocessing.train_split
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = create_default_config(
            model_name="test-model",
            gpu_arch=GPUArchitecture.ADA,
            vram_gb=16.0,
            compute_capability=8.9
        )
        
        assert valid_config.is_valid()
        assert len(valid_config.validate()) == 0
        
        # Invalid configuration (empty model name)
        invalid_config = create_default_config(
            model_name="",  # Empty name
            gpu_arch=GPUArchitecture.ADA,
            vram_gb=16.0,
            compute_capability=8.9
        )
        
        assert not invalid_config.is_valid()
        errors = invalid_config.validate()
        assert len(errors) > 0
        assert any("Model name cannot be empty" in error for error in errors)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_detect_model_info(self):
        """Test model information detection."""
        # Test Llama detection
        family, template = _detect_model_info("meta-llama/Llama-2-7b-chat")
        assert family == "llama"
        assert template == ChatTemplate.LLAMA
        
        # Test Gemma detection
        family, template = _detect_model_info("google/gemma-7b-it")
        assert family == "gemma"
        assert template == ChatTemplate.GEMMA
        
        # Test GPT detection
        family, template = _detect_model_info("microsoft/DialoGPT-medium")
        assert family == "gpt"
        assert template == ChatTemplate.CHATML
        
        # Test unknown model (fallback)
        family, template = _detect_model_info("unknown/model")
        assert family is None
        assert template == ChatTemplate.CHATML
    
    def test_calculate_batch_size(self):
        """Test batch size calculation."""
        # High VRAM
        assert _calculate_batch_size(24.0, 9.0) == 32  # Updated for aggressive batching
        
        # Medium VRAM
        assert _calculate_batch_size(16.0, 8.9) == 24  # Updated for RTX 4080/5080
        
        # Low VRAM
        assert _calculate_batch_size(8.0, 8.6) == 8   # Updated for aggressive batching
        
        # Very low VRAM
        assert _calculate_batch_size(4.0, 7.5) == 4   # Updated minimum
    
    def test_create_default_config(self):
        """Test default configuration creation."""
        config = create_default_config(
            model_name="microsoft/DialoGPT-medium",
            gpu_arch=GPUArchitecture.ADA,
            vram_gb=16.0,
            compute_capability=8.9
        )
        
        assert config.model.name == "microsoft/DialoGPT-medium"
        assert config.model.chat_template == ChatTemplate.CHATML
        assert config.hardware.gpu_arch == GPUArchitecture.ADA
        assert config.hardware.vram_gb == 16.0
        assert config.hardware.recommended_batch_size == 24  # Updated for consistency
        assert config.preprocessing.train_split == 0.9
    
    def test_validate_config_file(self):
        """Test configuration file validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            # Create and save valid configuration
            valid_config = create_default_config(
                model_name="test-model",
                gpu_arch=GPUArchitecture.ADA,
                vram_gb=16.0,
                compute_capability=8.9
            )
            valid_config.save(config_path)
            
            # Validate file
            is_valid, errors = validate_config_file(config_path)
            assert is_valid
            assert len(errors) == 0
            
            # Test non-existent file
            nonexistent_path = Path(temp_dir) / "nonexistent.yaml"
            is_valid, errors = validate_config_file(nonexistent_path)
            assert not is_valid
            assert len(errors) > 0


# Property-based tests would go here using hypothesis
# For now, we'll add some basic property tests

class TestConfigProperties:
    """Property-based tests for configuration system."""
    
    def test_config_roundtrip_property(self):
        """Property: Configuration should survive serialization roundtrip."""
        # Test with various valid configurations
        test_cases = [
            {
                "model_name": "microsoft/DialoGPT-medium",
                "gpu_arch": GPUArchitecture.ADA,
                "vram_gb": 16.0,
                "compute_capability": 8.9
            },
            {
                "model_name": "meta-llama/Llama-2-7b",
                "gpu_arch": GPUArchitecture.BLACKWELL,
                "vram_gb": 24.0,
                "compute_capability": 12.0
            },
            {
                "model_name": "google/gemma-7b",
                "gpu_arch": GPUArchitecture.AMPERE,
                "vram_gb": 12.0,
                "compute_capability": 8.6
            }
        ]
        
        for case in test_cases:
            original = create_default_config(**case)
            
            # Roundtrip through dictionary
            config_dict = original.to_dict()
            restored = ForgeConfig.from_dict(config_dict)
            
            # All fields should be preserved
            assert restored.model.name == original.model.name
            assert restored.model.chat_template == original.model.chat_template
            assert restored.hardware.gpu_arch == original.hardware.gpu_arch
            assert restored.hardware.vram_gb == original.hardware.vram_gb
            assert restored.preprocessing.train_split == original.preprocessing.train_split
    
    def test_validation_consistency_property(self):
        """Property: Valid configurations should always pass validation."""
        test_configs = [
            create_default_config("test-model-1", GPUArchitecture.ADA, 16.0, 8.9),
            create_default_config("test-model-2", GPUArchitecture.BLACKWELL, 24.0, 12.0),
            create_default_config("test-model-3", GPUArchitecture.AMPERE, 12.0, 8.6),
        ]
        
        for config in test_configs:
            assert config.is_valid(), f"Default config should be valid: {config.validate()}"
            assert len(config.validate()) == 0