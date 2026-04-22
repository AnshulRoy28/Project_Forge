# Security

## API Key Security

### Storage

API keys are stored using the system keyring:
- **Windows**: Windows Credential Manager (encrypted)
- **macOS**: Keychain (encrypted)
- **Linux**: Secret Service API (encrypted)

### Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for CI/CD
3. **Rotate keys periodically**
4. **Revoke compromised keys** immediately at [Google AI Studio](https://makersuite.google.com/app/apikey)
5. **Use separate keys** for different environments

### What We Do

✅ Store keys in encrypted system keyring  
✅ Hide key input when typing  
✅ Validate key format before storage  
✅ Test keys before accepting them  
✅ Never log keys (even in debug mode)  

### What We Don't Do

❌ Store keys in plain text files  
❌ Send keys anywhere except Google's API  
❌ Share keys with third parties  
❌ Include keys in error messages  

## Data Security

### Project Data

- All project data stays on your machine
- Data is stored in `.nnb/<project-id>/` directories
- No data is sent to external services except:
  - Project descriptions to Gemini API (for analysis)
  - Data samples to Gemini API (for validation)

### Docker Isolation

- Each project runs in an isolated Docker container
- Data is mounted read-only to prevent modification
- Containers have no network access during training (optional)
- Resource limits prevent resource exhaustion

## Reporting Security Issues

If you discover a security vulnerability:

1. **Do NOT** open a public issue
2. Email security@example.com (replace with actual email)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will respond within 48 hours.

## Security Updates

- Check for updates regularly: `pip install --upgrade nnb`
- Subscribe to security advisories on GitHub
- Review CHANGELOG.md for security fixes

---

Last updated: 2026-04-23
