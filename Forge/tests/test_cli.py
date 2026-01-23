"""
Tests for Forge CLI commands.
"""

import pytest
from typer.testing import CliRunner
from pathlib import Path

from forge.cli.main import app


runner = CliRunner()


class TestCLI:
    """Test CLI commands."""
    
    def test_version(self):
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "Forge CLI" in result.stdout
    
    def test_help(self):
        """Test --help flag."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "forge" in result.stdout.lower()
        assert "init" in result.stdout
        assert "study" in result.stdout
        assert "plan" in result.stdout
        assert "train" in result.stdout
        assert "test" in result.stdout
    
    def test_init_help(self):
        """Test init command help."""
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        assert "Initialize" in result.stdout or "API" in result.stdout
    
    def test_study_no_path(self):
        """Test study command without path shows error."""
        result = runner.invoke(app, ["study"])
        assert result.exit_code != 0  # Should fail without path
    
    def test_plan_help(self):
        """Test plan command help."""
        result = runner.invoke(app, ["plan", "--help"])
        assert result.exit_code == 0
        assert "goal" in result.stdout.lower() or "natural language" in result.stdout.lower()
    
    def test_train_no_config(self):
        """Test train command without config."""
        result = runner.invoke(app, ["train", "--dry-run"])
        # Should fail gracefully without config
        assert "not found" in result.stdout.lower() or result.exit_code != 0


class TestHardware:
    """Test hardware detection."""
    
    def test_detect_hardware(self):
        """Test hardware detection runs without errors."""
        from forge.core.hardware import detect_hardware
        
        profile = detect_hardware()
        
        assert profile is not None
        assert profile.system is not None
        assert profile.system.ram_total_gb > 0
        assert profile.system.cpu_cores > 0
    
    def test_recommend_config(self):
        """Test config recommendation."""
        from forge.core.hardware import detect_hardware
        
        profile = detect_hardware()
        config = profile.recommend_config()
        
        assert config is not None
        assert config.batch_size >= 1
        assert config.lora_rank >= 1


class TestConfig:
    """Test configuration handling."""
    
    def test_forge_config_defaults(self):
        """Test ForgeConfig has sensible defaults."""
        from forge.core.config import ForgeConfig
        
        config = ForgeConfig()
        
        assert config.training.batch_size >= 1
        assert config.training.learning_rate > 0
        assert config.training.num_epochs >= 1
    
    def test_config_serialization(self, tmp_path):
        """Test config can be saved and loaded."""
        from forge.core.config import ForgeConfig, save_config, load_config
        
        config = ForgeConfig(name="test-project", goal="Test training")
        config_path = tmp_path / "forge.yaml"
        
        save_config(config, config_path)
        loaded = load_config(config_path)
        
        assert loaded.name == "test-project"
        assert loaded.goal == "Test training"


class TestSecurity:
    """Test security features."""
    
    def test_script_analysis_safe(self):
        """Test safe script passes analysis."""
        from forge.core.security import analyze_script
        
        safe_script = '''
import torch
model = torch.nn.Linear(10, 5)
'''
        report = analyze_script(safe_script)
        
        assert report.is_safe
        assert report.risk_level in ("low", "medium")
    
    def test_script_analysis_dangerous(self):
        """Test dangerous script is flagged."""
        from forge.core.security import analyze_script
        
        dangerous_script = '''
import os
os.system("rm -rf /")
eval(user_input)
'''
        report = analyze_script(dangerous_script)
        
        assert not report.is_safe
        assert report.risk_level in ("high", "critical")
        assert len(report.issues) > 0
    
    def test_api_key_validation(self):
        """Test API key format validation."""
        from forge.core.security import validate_api_key_format
        
        assert not validate_api_key_format("")
        assert not validate_api_key_format("short")
        assert validate_api_key_format("AIzaSyC" + "x" * 30)  # Valid format
