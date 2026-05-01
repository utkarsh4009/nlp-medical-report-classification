# Contributing to Medical Transcription NLP Classification

Thank you for your interest in contributing! Here are the guidelines to help you get started.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (Python version, OS, library versions)

### Suggesting Enhancements

Open an issue with:
- A clear description of the enhancement
- Why it would be useful for this project
- Any relevant references or resources

### Pull Requests

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** — ensure they follow the code style (PEP 8)
4. **Test your changes** by running the notebook end-to-end
5. **Commit** with a meaningful message:
   ```bash
   git commit -m "feat: add ClinicalBERT fine-tuning option"
   ```
6. **Push** to your fork and **open a Pull Request**

## Commit Message Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | When to use |
|--------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation changes |
| `refactor:` | Code refactoring |
| `perf:` | Performance improvement |
| `test:` | Adding/updating tests |
| `chore:` | Maintenance tasks |

## Development Setup

```bash
git clone https://github.com/Rishijain411/nlp-medical-classification.git
cd nlp-medical-classification
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Code Style

- Follow **PEP 8** for Python code
- Use **descriptive variable names** (especially for clinical terms)
- **Comment** all NLP-specific decisions (why a particular regex, why a specific stopword list)
- Keep functions **focused** — one function, one responsibility

## Areas Open for Contribution

- [ ] ClinicalBERT / BioBERT integration
- [ ] UMLS-based abbreviation expansion
- [ ] NegEx / NegBio negation detection
- [ ] REST API wrapper for the triage classifier
- [ ] Streamlit web demo
- [ ] Unit tests for preprocessing functions
- [ ] Support for additional medical specialties

## Questions?

Open an issue tagged `question` — we're happy to help!
