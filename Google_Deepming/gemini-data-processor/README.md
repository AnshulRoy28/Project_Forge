# Gemini Data Processor

An AI-powered command-line tool that leverages Google's Gemini AI to intelligently analyze and process datasets through modular script execution in isolated Docker containers.

## Features

- **AI-Powered Analysis**: Uses Gemini AI to automatically analyze data structure, detect quality issues, and suggest processing operations
- **Docker Isolation**: All data processing scripts run in isolated Docker containers, keeping your local environment clean
- **Human-in-the-Loop**: Every operation requires explicit user approval before execution
- **Modular Processing**: Processing is broken into small, focused scripts rather than monolithic operations
- **Context Awareness**: Maintains processing history to inform subsequent operations
- **Multi-Format Support**: Supports CSV, JSON, and text file formats

## Installation

### Prerequisites

- Python 3.8 or higher
- Docker 20.0 or higher (installed and running)
- Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### Install from source

```bash
cd gemini-data-processor
pip install -e .
```

### Install for development

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Initialize with your Gemini API key

```bash
gdp init
```

This will prompt you to enter your Gemini API key, which is securely stored in your system keyring.

### 2. Check system status

```bash
gdp status
```

Verify that Docker is running and your configuration is correct.

### 3. Process a data file

```bash
gdp process your_data.csv
```

The tool will:
1. Analyze your data structure
2. Identify quality issues
3. Generate processing scripts
4. Request approval before executing each script
5. Save processed output to a timestamped directory

## Commands

### `gdp init`

Initialize the tool with your Gemini API key.

```bash
gdp init                    # Interactive prompts
gdp init --api-key YOUR_KEY # Direct key input
gdp init --force            # Overwrite existing config
```

### `gdp status`

Check the status of your configuration and Docker environment.

```bash
gdp status
```

### `gdp process`

Process a data file using Gemini AI.

```bash
gdp process data.csv                    # Process with default settings
gdp process data.csv -o ./output        # Specify output directory
gdp process data.csv --dry-run          # Show analysis without executing
gdp process data.csv -v                 # Verbose output
```

### `gdp clean`

Clean up temporary files and old sessions.

```bash
gdp clean          # Show cleanup options
gdp clean --all    # Clear all sessions and cache
```

## Configuration

The tool stores configuration in `.gemini-processor/` within your project directory:

```
.gemini-processor/
├── .gemini-processor.json  # Configuration settings
├── context.db              # SQLite database for session history
└── sessions/               # Session working directories
```

### Environment Variables

- `GEMINI_API_KEY`: Your Gemini API key (alternative to keyring storage)
- `GEMINI_MODEL`: Model to use (default: `gemini-2.0-flash`)
- `GEMINI_MAX_TOKENS`: Maximum tokens for responses
- `GDP_MAX_MEMORY_GB`: Container memory limit (default: 4)
- `GDP_MAX_CPU_CORES`: Container CPU limit (default: 2)

## Security

- **API Key Security**: Keys are stored in your system keyring or environment variables, never in plain text files
- **Container Isolation**: All scripts run in isolated Docker containers with:
  - No network access by default
  - Read-only access to input data
  - Resource limits (CPU, memory, disk)
  - Non-root execution
- **Script Validation**: All generated scripts are validated for syntax and checked for potentially dangerous operations
- **Human Approval**: Every script execution requires explicit user approval

## Resource Limits

Default container limits:
- **Memory**: 4 GB
- **CPU**: 2 cores
- **Disk**: 10 GB
- **Execution Time**: 30 minutes

## Supported File Formats

- **CSV**: Comma-separated values with header detection
- **JSON**: Objects, arrays, or nested structures
- **Text**: Line-based text files

## Architecture

```
┌───────────────────┐
│   CLI Interface   │
└─────────┬─────────┘
          │
┌─────────▼─────────┐
│ Processing Engine │
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    │           │
┌───▼───┐   ┌───▼───┐
│Gemini │   │Docker │
│  AI   │   │Manager│
└───────┘   └───────┘
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/
ruff check src/
```

### Type Checking

```bash
mypy src/
```

## License

MIT License
