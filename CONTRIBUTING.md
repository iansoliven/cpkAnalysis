# Contributing to CPK Analysis

Thank you for contributing to CPK Analysis! This guide will help you get started.

## Quick Start for Developers

### 1. Setup Development Environment

```bash
# Clone with submodules
git clone --recursive https://remote.example.com/cpkAnalysis.git
cd cpkAnalysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest pytest-cov black ruff mypy

# Verify installation
pytest
```

### 2. Before Submitting Code

**Required checks (all must pass):**

```bash
# 1. Run all tests
pytest

# 2. Check coverage (aim for 75%+ overall)
pytest --cov=cpkanalysis --cov-report=html:test_coverage_report

# 3. Format code
black cpkanalysis/ tests/

# 4. Lint code
ruff check cpkanalysis/ tests/
```

### 3. Pull Request Checklist

- [ ] All tests pass (`pytest`)
- [ ] Code coverage maintained or improved
- [ ] Code formatted with `black`
- [ ] No linting errors (`ruff check`)
- [ ] Tests added for new features/bug fixes
- [ ] Documentation updated (README, help files, docstrings)
- [ ] Descriptive commit messages
- [ ] PR description explains changes and motivation

## Code Standards

### Testing Requirements

**All code contributions MUST include tests.**

- **New features:** Unit tests + integration tests
- **Bug fixes:** Regression test demonstrating the bug, then fix
- **Edge cases:** Tests for NaN, Inf, empty data, boundary conditions

### Code Style

- Follow **PEP 8** style guide
- Use **type hints** on all function signatures
- Use **dataclasses** for data structures
- Prefer `pathlib.Path` over `os.path`
- Use f-strings for formatting

### Test Style

- **Arrange-Act-Assert** pattern
- One logical assertion per test
- Use fixtures for common setup
- Parametrize for multiple scenarios
- Mock external dependencies

## Commit Message Format

Use conventional commits format:

```
<type>: <description>

[optional body]
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `test:` Add/update tests
- `docs:` Documentation changes
- `refactor:` Code restructuring
- `perf:` Performance improvement
- `chore:` Maintenance tasks

**Examples:**
```
feat: Add Spec/What-If limit update action
fix: Preserve OPT_FLAG bit 4/5 defaults correctly
test: Add comprehensive CPK calculation tests
docs: Update testing guidance with new test files
```

## Documentation

### Comprehensive Testing Guide

**For detailed information on testing, see:**

📖 **[help/testing_guidance.html](help/testing_guidance.html)**

This comprehensive guide covers:
- Current test suite status (90 tests, 75% coverage)
- All 17 test files with descriptions
- Module coverage statistics
- How to run tests (basic, coverage, parallel)
- Writing new tests (naming, patterns, fixtures)
- CI/CD integration (remote actions, GitLab CI)
- Debugging test failures
- Advanced techniques (mocking, property-based testing)
- Pull request requirements

### Other Documentation

- **[README.md](README.md)** — Project overview and quick start
- **[help/getting_started.html](help/getting_started.html)** — Installation guide
- **[help/cli_reference.html](help/cli_reference.html)** — Command-line options
- **[help/stdf_ingestion.html](help/stdf_ingestion.html)** — STDF technical reference
- **[help/post_processing.html](help/post_processing.html)** — Post-processing guide
- **[help/manual_verification.html](help/manual_verification.html)** — QA procedures

## Project Structure

```
cpkAnalysis/
├── cpkanalysis/              # Main package
│   ├── __init__.py
│   ├── cli.py                # CLI interface
│   ├── gui.py                # Console GUI
│   ├── ingest.py             # STDF ingestion
│   ├── pipeline.py           # Pipeline orchestration
│   ├── stats.py              # Statistical calculations
│   ├── mpl_charts.py         # Chart rendering
│   ├── workbook_builder.py   # Excel generation
│   ├── plugins.py            # Plugin system
│   └── postprocess/          # Post-processing subsystem
├── tests/                    # Test suite (90 tests, 75% coverage)
│   ├── test_cli_commands.py
│   ├── test_ingest_*.py
│   ├── test_stats_*.py
│   ├── test_postprocess*.py
│   └── ...
├── help/                     # HTML documentation
├── Sample/                   # Sample STDF files
├── README.md                 # Main documentation
├── CONTRIBUTING.md           # This file
└── pyproject.toml            # Project configuration
```

## Test Coverage Goals

| Component | Target Coverage | Priority |
|-----------|-----------------|----------|
| STDF Ingestion | 85%+ | Critical |
| Statistics & CPK | 85%+ | Critical |
| Post-Processing | 80%+ | High |
| Pipeline | 85%+ | High |
| Plugin System | 80%+ | High |
| CLI Commands | 70%+ | Medium |
| Chart Generation | 75%+ | Medium |

## Quality Standards

### What We Value

- **Reliability:** Tests must pass consistently
- **Performance:** Unit tests < 100ms, integration tests < 5s
- **Maintainability:** Clear code, good documentation
- **Compatibility:** Python 3.11+ on Windows/macOS/Linux

### What to Avoid

- ❌ Flaky tests (random failures)
- ❌ Tests that depend on execution order
- ❌ Duplicate test code (use fixtures)
- ❌ Vague test names
- ❌ Tests without assertions
- ❌ Submitting code without tests

## Getting Help

- 📖 **Documentation:** [help/index.html](help/index.html)
- 🐛 **Issues:** [Issue Tracker](https://remote.example.com/cpkAnalysis/issues)
- 💬 **Discussions:** [Project Discussions](https://remote.example.com/cpkAnalysis/discussions)

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers this project.

---

**Thank you for contributing to CPK Analysis!** 🎉

