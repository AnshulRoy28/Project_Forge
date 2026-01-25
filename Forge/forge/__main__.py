#!/usr/bin/env python3
"""
Forge CLI - Main module entry point.

This allows running the CLI as: python -m forge
"""

def main():
    """Entry point for the forge CLI."""
    from forge.cli.main import main as cli_main
    cli_main()

if __name__ == "__main__":
    main()