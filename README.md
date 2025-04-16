# LLMQA

LLMQA (Large Language Model Question and Answer) is a Python library for generating and evaluating question-answer pairs from text chunks using large language models. It is designed to create custom validation datasets for evaluating RAG LLM systems against base LLMs.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [License](#license)

## Introduction

LLMQA provides a unified interface for working with multiple LLM providers and generating high-quality question-answer pairs from chunked text. The system provides:

1. Multi-Provider Support: Work with various LLM providers including:
   - Databricks
   - Google
   - OpenRouter
   - ROSIE

2. QA Generation: Generate QA pairs from text chunks using the `LLMService` interface
3. Quality Evaluation: Evaluate QA pairs using the `CritiqueEvaluator` with configurable criteria
4. LLM Evaluation: Evaluate a model (RAG or not) on generated QA pairs with LLM-as-a-judge techniques to quanitfy performance on individualized validation datasets

## Installation

### Package Manager

LLMQA uses `uv` as its package manager. It is currently only available on TestPyPI in pre-release form. This will be updated
when it is available on PyPI.

### Environment Variables

The following environment variables are required for each LLM provider you wish to use:

```bash
# Databricks
export DATABRICKS_KEY='<your-api-key>'
export DATABRICKS_URL='<your-endpoint-url>'

# Google
export GOOGLE_KEY='<your-api-key>'
export GOOGLE_URL='<your-endpoint-url>'

# OpenRouter
export OPENROUTER_KEY='<your-api-key>'
export OPENROUTER_URL='<your-endpoint-url>'

# ROSIE
export ROSIE_KEY='<your-api-key>'
export ROSIE_URL='<your-endpoint-url>'
```

## Usage

### Python API

```python
from llmqa.core.llm_service import LLMService
from llmqa.models.databricks import DatabricksModel, ROSIELlama
from llmqa.evaluators.critique import CritiqueEvaluator

# Initialize models
gen_model = ROSIELlama()
critique_model = DatabricksModel()
evaluator = CritiqueEvaluator(critique_model)

# Create LLM service
service = LLMService(model=gen_model, evaluator=evaluator)

# Generate and evaluate QA pairs
text_chunk = "Your text chunk here..."
qa_pairs = service.generate_qa(
    text_chunk=text_chunk,
    with_critique=True,
    criteria=['groundedness', 'relevance', 'standalone'] #TODO fix criteria example
)
```

### Logging Configuration

LLMQA provides a comprehensive logging system for debugging and monitoring:

```python
from llmqa.utils.logging_config import setup_logging

# Set up logging with debug level
logger = setup_logging(level=logging.DEBUG, verbose=True)

# Use the logger in your code
logger.debug("Debug message")
logger.info("Info message")
```

## Directory Structure

```
LLMQA/
├── llmqa/                      # Main package directory
│   ├── __init__.py            # Package initialization
│   ├── core/                  # Core functionality
│   │   └── llm_service.py     # Main LLM service interface
│   ├── models/                # LLM model implementations
│   │   ├── __init__.py
│   │   ├── base.py           # Base model class
│   │   ├── databricks.py     # Databricks implementation
│   │   ├── google.py         # Google implementation
│   │   ├── openrouter.py     # OpenRouter implementation
│   │   └── rosie_llama.py    # ROSIE implementation
│   ├── evaluators/           # Evaluation implementations
│   │   ├── __init__.py
│   │   └── critique.py       # Critique evaluation logic
│   └── utils/                # Utility functions
│       └── logging_config.py # Logging configuration
├── tests/                    # Test files
├── pyproject.toml           # Project configuration
├── uv.lock                  # Package lock file
└── README.md                # Documentation
```

## License

Open Source (OSI Approved): [MIT License](LICENSE)
