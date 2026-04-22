# 🚀 Run Me First - Simple Setup

## Step 1: Open PowerShell

1. Press `Windows + X`
2. Click "Windows PowerShell" or "Terminal"

## Step 2: Navigate to Project

```powershell
cd "D:\Programming\Project Forge"
```

## Step 3: Check Python

```powershell
python --version
```

You should see: `Python 3.11.0` (or similar)

## Step 4: Install the Tool

```powershell
pip install -e .
```

Wait for it to finish (takes ~30 seconds).

## Step 5: Verify Installation

```powershell
nnb --help
```

You should see a list of commands.

## Step 6: Get Your Gemini API Key

1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy the key (starts with `AIza...`)

## Step 7: Start Your First Project

```powershell
nnb start
```

The tool will:
1. Ask for your API key (paste it - you won't see it, that's normal)
2. Test the key
3. Store it securely
4. Start the conversation

## Step 8: Describe Your Project

Type something like:

```
I want to build an image classifier for cats and dogs.
I have 1000 images of each.
I need it to be reasonably accurate.
```

Then press **Enter twice** (two times).

## Step 9: Answer Questions

Gemini will ask questions like:
- "How many classes?" → Type: `2`
- "Input size?" → Type: `224x224`
- "Use pretrained model?" → Type: `yes`

## Step 10: Confirm

When asked "Does this look correct?", type: `y`

## Step 11: Validate Your Data

When you have your data ready:

```powershell
nnb data validate --path "C:\path\to\your\data"
```

Replace `C:\path\to\your\data` with your actual data folder.

## That's It! 🎉

The tool will guide you through the rest:
- Building Docker environment
- Generating code
- Training
- Inference

## Quick Commands

```powershell
# Check status
nnb status

# Check API key
nnb config status

# Get help
nnb --help
```

## Troubleshooting

### "nnb: command not found"

Run this again:
```powershell
pip install -e .
```

### "No API key configured"

Run this:
```powershell
nnb config setup
```

### Need to start over?

Just run:
```powershell
nnb start
```

It will create a new project.

---

**Questions? Check [QUICKSTART.md](QUICKSTART.md) for detailed guide.**
