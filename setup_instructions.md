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

# 4. Initialize Forge (configure API keys)
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

## Troubleshooting

If you get "neither 'setup.py' nor 'pyproject.toml' found":
- Make sure you're in the Forge directory: `cd Forge`
- Verify pyproject.toml exists: `ls pyproject.toml`
- The file should be at the root level of the cloned repository

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