# CLI Neural Network Builder - Kiro Configuration

This directory contains steering documents, hooks, and configuration for the CLI Neural Network Builder project.

## 📁 Directory Structure

```
.kiro/
├── steering/              # Steering documents (auto-included guidance)
│   ├── architecture.md
│   ├── coding-style.md
│   ├── docker-patterns.md
│   ├── error-handling.md
│   ├── gemini-prompts.md
│   ├── lessons-learned.md
│   ├── security.md
│   ├── stage-workflow.md
│   └── testing.md
└── README.md             # This file
```

## 📚 Steering Documents

Steering documents provide guidance to AI agents working on this project. All documents in the `steering/` directory are automatically included in agent context.

### Core Documents

| Document | Purpose |
|----------|---------|
| **architecture.md** | Three-layer architecture, state machine, container lifecycle |
| **coding-style.md** | Immutability, file organization, error handling, code quality |
| **docker-patterns.md** | Container management, volume mounts, health checks |
| **error-handling.md** | Error categories, recovery strategies, user communication |
| **gemini-prompts.md** | Prompt engineering patterns for each pipeline stage |
| **security.md** | API key management, container isolation, input validation |
| **stage-workflow.md** | Detailed workflow for each of the 9 pipeline stages |
| **testing.md** | TDD workflow, test coverage requirements, test patterns |
| **lessons-learned.md** | Project-specific patterns and insights (user-editable) |

### Document Format

Each steering document includes YAML frontmatter:

```yaml
---
inclusion: auto
description: Brief description of the document's purpose
---
```

- `inclusion: auto` — Document is automatically included in agent context
- `description` — Brief summary of what the document covers

## 🎯 Key Principles

### Architecture
- **Three layers**: Orchestrator (CLI), Docker Runtime, Gemini Brain
- **State machine**: 9 stages from INIT to DONE
- **One container per project**: Built once, reused throughout
- **Volume mounts**: Integration boundary between host and container

### Code Quality
- **Immutability**: Never mutate, always create new objects
- **Small files**: 200-400 lines typical, 800 max
- **Error handling**: Comprehensive with user-friendly messages
- **Testing**: 80% coverage minimum, TDD workflow

### Docker
- **Read-only data mounts**: Prevent accidental data modification
- **Detached execution**: Training runs in background
- **Health checks**: Validate container after build
- **Reproducibility**: Generated Dockerfile committed with project

### Security
- **No hardcoded secrets**: Use environment variables or keyring
- **Input validation**: Validate all user inputs
- **Container isolation**: Network isolation, resource limits
- **Sanitized logging**: Remove sensitive data from logs

## 🔧 Usage

### For AI Agents

When working on this project, agents should:

1. **Read relevant steering documents** before making changes
2. **Follow established patterns** from the steering documents
3. **Update lessons-learned.md** when discovering new patterns
4. **Validate against architecture principles** before implementing

### For Developers

When contributing to this project:

1. **Review steering documents** to understand project conventions
2. **Follow the patterns** described in the documents
3. **Update documents** when conventions change
4. **Add insights** to lessons-learned.md as you discover them

## 📝 Editing Steering Documents

### When to Edit

- **Architecture changes**: Update architecture.md
- **New patterns discovered**: Add to lessons-learned.md
- **Security requirements change**: Update security.md
- **Testing strategy evolves**: Update testing.md

### How to Edit

1. Open the relevant `.md` file in `steering/`
2. Make your changes
3. Ensure YAML frontmatter is preserved
4. Commit changes with descriptive message

### Adding New Documents

To add a new steering document:

1. Create `steering/your-document.md`
2. Add YAML frontmatter:
   ```yaml
   ---
   inclusion: auto
   description: Your description here
   ---
   ```
3. Write your content in Markdown
4. Document will be automatically included

## 🎓 Learning Resources

### Understanding the Pipeline

Start with these documents in order:

1. **architecture.md** — Understand the three-layer architecture
2. **stage-workflow.md** — Learn the 9-stage pipeline
3. **docker-patterns.md** — Understand container management
4. **gemini-prompts.md** — Learn how to interact with Gemini

### Writing Code

Reference these documents while coding:

1. **coding-style.md** — Code quality standards
2. **error-handling.md** — How to handle errors properly
3. **testing.md** — Testing requirements and patterns
4. **security.md** — Security best practices

### Debugging Issues

When troubleshooting:

1. **lessons-learned.md** — Check for known pitfalls
2. **error-handling.md** — Understand error recovery strategies
3. **docker-patterns.md** — Debug container issues
4. **stage-workflow.md** — Understand expected stage behavior

## 🔄 Maintenance

### Regular Updates

- **After major features**: Update relevant steering documents
- **When patterns emerge**: Add to lessons-learned.md
- **When conventions change**: Update affected documents
- **After incidents**: Document in lessons-learned.md

### Review Schedule

- **Monthly**: Review lessons-learned.md for outdated entries
- **Quarterly**: Review all documents for accuracy
- **After releases**: Update with any new patterns or changes

## 📞 Questions?

If you have questions about:

- **Architecture decisions**: See architecture.md
- **Coding standards**: See coding-style.md
- **Docker usage**: See docker-patterns.md
- **Testing approach**: See testing.md
- **Security concerns**: See security.md

For questions not covered in the steering documents, consult the project maintainers.

---

**Last Updated**: 2026-04-23
**Version**: 1.0.0
