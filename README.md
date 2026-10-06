# NOVA IMS Programming Methods 2026

Welcome to the Programming Methods repository for NOVA IMS. This project contains the course labs, notebooks, and supporting code for Python-based software engineering work.

## Project structure

```text
.
├── data/                 # Data files used across exercises
├── docs/                 # Documentation and setup guides
│   └── INSTALLATION.md   # Environment and installation instructions
├── notebooks/            # Jupyter notebooks for labs
├── src/                  # Source code for the course projects
│   └── pm_labs/          # Lab package and Python modules
├── tests/                # Automated tests
├── pyproject.toml        # Python project configuration and dependencies
├── README.md             # Project overview
└── .gitignore            # Ignores local environment and generated files
```

## Getting started

For environment setup and installation instructions, see [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Running Tests

From the repository root, run the full test suite with:

```bash
pytest
```

To run tests for one module, pass its test file, for example:

```bash
pytest tests/test_pipeline.py
```
