# Forge Setup Instructions

## Fresh Installation from GitHub

```bash
# 1. Clone the repository
git clone https://github.com/AnshulRoy28/Forge.git
cd Forge

# 2. Verify you're in the right directory (should see pyproject.toml)
ls pyproject.toml

# 3. Install Forge
pip install -e .

# 4. Initialize Forge (configure session credentials)
forge init

# 5. Build Docker container
forge docker build

# 6. Generate training plan
forge plan "Create a helpful chatbot assistant"

# 7. Prepare dataset (if you have data)
forge prepare ./data/your_data.csv

# 8. Start training
forge train
```

## Security Model

**🔒 Session-Based Security**: Forge uses session-based credential storage for maximum security:

- **No Persistent Storage**: API keys are stored only in memory for the current session
- **Automatic Cleanup**: Credentials are automatically cleared when the terminal closes
- **Fresh Start**: Each new terminal session requires re-entering credentials
- **Zero Risk**: No sensitive data persists on disk or across sessions

### Available Commands:
- `forge init` - Configure credentials for the current session
- `forge login` - Update or verify session credentials  
- `forge cleanup` - Manually clear session credentials
- `forge login --verify` - Check what credentials are available

## Troubleshooting

If you get "neither 'setup.py' nor 'pyproject.toml' found":
- Make sure you're in the Forge directory: `cd Forge`
- Verify pyproject.toml exists: `ls pyproject.toml`
- The file should be at the root level of the cloned repository

If you get "Session credentials required":
- Run `forge init` to configure API keys for the current session
- Or run `forge login` to update existing session credentials

## Repository Structure
```
Forge/
├── pyproject.toml          # Python package configuration
├── forge/                  # Main package
├── docker/                 # Docker containers
├── tests/                  # Test files
├── README.md
└── ...
```

## What Each Command Does

- `forge init`: Configures session API keys (Gemini + HuggingFace) and detects hardware
- `forge docker build`: Builds optimized Docker container for your GPU
- `forge plan`: Uses Gemini AI to generate training configuration
- `forge prepare`: Preprocesses your dataset for training
- `forge train`: Starts Docker-based training with real-time progress
- `forge cleanup`: Clears session credentials and temporary files

## Security Benefits

✅ **No Keyring Dependencies**: Removed system keyring storage  
✅ **Session Isolation**: Each terminal session is independent  
✅ **Automatic Cleanup**: Credentials cleared on session end  
✅ **Zero Persistence**: No sensitive data stored on disk  
✅ **Fresh Authentication**: Always requires explicit credential entry