# API Key Setup Guide

## Overview

The CLI Neural Network Builder requires a Gemini API key to function. Your API key is stored **securely** in your system keyring (the same place your browser stores passwords), never in plain text files.

## Getting Your API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key (starts with `AIza...`)

## Setup Methods

### Method 1: Interactive Setup (Recommended)

The easiest way is to let the tool guide you:

```bash
# Run the setup command
nnb config setup
```

The tool will:
1. Prompt you to enter your API key (input is hidden)
2. Validate the key format
3. Test the key with a simple API call
4. Store it securely in your system keyring

### Method 2: Automatic Setup on First Use

When you run `nnb start` for the first time without a configured key:

```bash
nnb start
```

The tool will automatically prompt you to set up your API key before proceeding.

### Method 3: Environment Variable (Advanced)

For CI/CD, Docker, or temporary use:

```bash
# Linux/Mac
export GEMINI_API_KEY=your-key-here

# Windows PowerShell
$env:GEMINI_API_KEY="your-key-here"

# Windows CMD
set GEMINI_API_KEY=your-key-here
```

**Note**: Environment variables take precedence over keyring storage.

## Managing Your API Key

### Check Status

See if your API key is configured:

```bash
nnb config status
```

Output:
```
🔑 API Key Status

✓ API key found in system keyring
  Key: AIzaSyC1234...st
```

### Update Key

To update your existing key:

```bash
nnb config setup
```

When prompted "Do you want to update it?", choose Yes.

### Delete Key

To remove your API key from the system:

```bash
nnb config delete-key
```

You'll be asked to confirm before deletion.

## Security Features

### System Keyring Storage

Your API key is stored in your operating system's secure credential storage:

- **Windows**: Windows Credential Manager
- **macOS**: Keychain
- **Linux**: Secret Service API (GNOME Keyring, KWallet, etc.)

This is the same system used by:
- Your web browser for saved passwords
- Password managers
- Other secure applications

### Key Protection

✅ **Encrypted at rest** - Your OS encrypts the keyring  
✅ **Never in plain text** - Not stored in config files or environment  
✅ **Hidden input** - Key is not displayed when typing  
✅ **Validation** - Format checked before storage  
✅ **Testing** - Key is tested with API before storage  

### What We DON'T Do

❌ We never send your key anywhere except Google's Gemini API  
❌ We never log your key (even in debug logs)  
❌ We never store it in plain text files  
❌ We never share it with third parties  

## Troubleshooting

### "Could not access keyring"

**Problem**: The system keyring is not available or accessible.

**Solutions**:
1. **Linux**: Install a keyring backend:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install gnome-keyring
   
   # Fedora
   sudo dnf install gnome-keyring
   ```

2. **Use environment variable** as a fallback:
   ```bash
   export GEMINI_API_KEY=your-key-here
   ```

### "API key test failed"

**Problem**: The key format is correct but API calls fail.

**Possible causes**:
1. **Invalid key** - Double-check you copied the entire key
2. **Network issue** - Check your internet connection
3. **API quota** - Check your Google AI Studio quota
4. **Firewall** - Ensure access to `generativelanguage.googleapis.com`

**Solution**:
```bash
# Try again with a fresh key
nnb config setup
```

### "API key seems too short"

**Problem**: The key you entered is shorter than expected.

**Solution**: Gemini API keys are typically 39 characters. Make sure you copied the entire key from Google AI Studio.

### Key Not Found After Setup

**Problem**: Setup succeeded but key is not found later.

**Possible causes**:
1. **Different user account** - Keyring is per-user
2. **Keyring backend changed** - Linux only

**Solution**:
```bash
# Check status
nnb config status

# Re-run setup if needed
nnb config setup
```

## Best Practices

### For Development

✅ Use `nnb config setup` for local development  
✅ Keep your key private (never commit to git)  
✅ Use `.gitignore` to exclude any key files  

### For CI/CD

✅ Use environment variables in CI/CD pipelines  
✅ Store keys in your CI/CD platform's secrets manager  
✅ Never hardcode keys in scripts  

Example GitHub Actions:
```yaml
- name: Run tests
  env:
    GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
  run: |
    nnb start
```

### For Docker

✅ Pass key as environment variable:
```bash
docker run -e GEMINI_API_KEY=your-key-here your-image
```

✅ Or use Docker secrets:
```bash
docker run --env-file .env your-image
```

### For Teams

✅ Each team member should have their own API key  
✅ Use separate keys for dev/staging/production  
✅ Rotate keys periodically  
✅ Revoke keys when team members leave  

## FAQ

### Q: Where exactly is my key stored?

**A**: On Windows, it's in Windows Credential Manager. You can view it:
1. Open Control Panel
2. Search for "Credential Manager"
3. Look under "Generic Credentials" for "nnb-cli"

On macOS, it's in Keychain Access under "nnb-cli".

### Q: Can I use the same key on multiple machines?

**A**: Yes! You can set up the same API key on as many machines as you want. Each machine stores it independently in its own keyring.

### Q: What happens if I delete my key?

**A**: The tool will prompt you to set it up again the next time you try to use it. Your projects and data are not affected.

### Q: Can I have different keys for different projects?

**A**: Currently, the tool uses one key for all projects. If you need different keys, use environment variables:

```bash
# Project 1
GEMINI_API_KEY=key1 nnb start

# Project 2
GEMINI_API_KEY=key2 nnb start
```

### Q: Is my key sent anywhere besides Google?

**A**: No. Your key is only sent to Google's Gemini API endpoints. We never send it anywhere else.

### Q: How do I know if my key is working?

**A**: Run `nnb config status` to check if it's configured, or try starting a project with `nnb start` - it will test the key automatically.

## Support

If you encounter issues not covered here:

1. Check the [main README](../README.md)
2. Run `nnb config status` for diagnostics
3. Check the logs in `.nnb/logs/`
4. Open an issue on GitHub

---

**Security Note**: Never share your API key publicly or commit it to version control. Treat it like a password.
