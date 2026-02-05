"""Gemini Data Processor - AI-powered data analysis and processing CLI tool."""

__version__ = "0.1.0"
__author__ = "Gemini Data Processor Team"

# Re-export main CLI entry point
from .cli import main

__all__ = ["main", "__version__"]
