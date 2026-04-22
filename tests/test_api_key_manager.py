"""Test API key management."""

import pytest
import os
from unittest.mock import patch, MagicMock

from nnb.utils.api_key_manager import APIKeyManager


def test_get_api_key_from_environment():
    """Test getting API key from environment variable."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-from-env"}):
        key = APIKeyManager.get_api_key()
        assert key == "test-key-from-env"


def test_validate_api_key_valid():
    """Test validating a valid API key."""
    valid_key = "AIzaSyC1234567890abcdefghijklmnopqrst"
    assert APIKeyManager.validate_api_key(valid_key) is True


def test_validate_api_key_too_short():
    """Test validating a too-short API key."""
    short_key = "short"
    assert APIKeyManager.validate_api_key(short_key) is False


def test_validate_api_key_empty():
    """Test validating an empty API key."""
    assert APIKeyManager.validate_api_key("") is False
    assert APIKeyManager.validate_api_key(None) is False


def test_has_api_key_with_env():
    """Test checking if API key exists via environment."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        assert APIKeyManager.has_api_key() is True


def test_has_api_key_without_env():
    """Test checking if API key exists without environment."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("keyring.get_password", return_value=None):
            assert APIKeyManager.has_api_key() is False


@patch("keyring.set_password")
def test_set_api_key(mock_set_password):
    """Test setting API key."""
    mock_set_password.return_value = None
    
    result = APIKeyManager.set_api_key("test-key")
    
    assert result is True
    mock_set_password.assert_called_once_with("nnb-cli", "gemini_api_key", "test-key")


@patch("keyring.delete_password")
def test_delete_api_key(mock_delete_password):
    """Test deleting API key."""
    mock_delete_password.return_value = None
    
    result = APIKeyManager.delete_api_key()
    
    assert result is True
    mock_delete_password.assert_called_once_with("nnb-cli", "gemini_api_key")
