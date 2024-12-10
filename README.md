# LLMQA

LLMQA (Large Language Model Question and Answer) is a Python library for generating and evaluating question-answer pairs from text chunks using large language models.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [License](#license)

## Introduction

LLMQA is designed to generate and evaluate question-answer pairs from chunked text for use in evaluation of RAG LLM systems. The system provides:

1. QA Generation: Generates QA pairs from text chunks using LLMs
2. Quality Evaluation: Evaluates QA pairs using multiple criteria:
   - Groundedness: How well the question can be answered from the context
   - Relevance: How useful the question is for the environment in which it would be used
   - Standalone: How context-independent the question is
3. Parallel Processing: Efficient processing of large datasets
4. Quality Filtering: Automatic filtering of low-quality QA pairs

## Installation

### Conda Environment

Create the conda environment:

```bash
conda env create -f environment.yml
```

### Environment Variables

Add the following environment variables to your conda environment:

```bash
cd <path to conda environment>
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
```

In `./etc/conda/activate.d/env_vars.sh`:
```bash
#!/bin/sh

export ROSIE_URL='<URL to ROSIE LLM>'
export ROSIE_KEY='<ROSIE API_KEY>'
```

In `./etc/conda/deactivate.d/env_vars.sh`:
```bash
#!/bin/sh

unset ROSIE_URL
unset ROSIE_KEY
```

### Python Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### Python API

```python
from llmqa import ROSIELlama, QAGenerator, CritiqueAgent

# Initialize models
qa_model = ROSIELlama()
critique_model = ROSIELlama()
critique_agent = CritiqueAgent(critique_model)

# Create generator with critique
generator = QAGenerator(
    model=qa_model,
    critique_agent=critique_agent,
    min_critique_score=3.0,
    critique_criteria=['groundedness', 'relevance', 'standalone']
)

# Generate and evaluate QA pairs
qa_pairs = generator.generate_from_file(
    input_file="chunks/input.csv",
    output_file="output.json"
)
```

### Batch Processing

1. Use the template batch script:
```bash
cp scripts/batch_llmqa.sh scripts/examples/my_job.sh
```

2. Edit the configuration:
```bash
INPUT_FILE="path/to/chunks.csv"
OUTPUT_FILE="path/to/output.json"
```

3. Submit the job:
```bash
sbatch scripts/examples/my_job.sh
```

### Debugging

For testing and debugging, use the debug script:
```bash
python -m tests.debug_qa_critique
```

## Directory Structure

```
LLMQA/
├── llmqa/                      # Main package directory
│   ├── __init__.py            # Package initialization
│   ├── models/                # LLM model implementations
│   │   ├── __init__.py
│   │   ├── base.py           # Base model class
│   │   ├── rosie_llama.py    # ROSIE implementation
│   │   └── critique_agent.py  # QA evaluation agent
│   ├── generators/           # QA generation implementations
│   │   ├── __init__.py
│   │   └── qa_generator.py   # QA generation logic
│   └── utils/                # Utility functions
│       └── __init__.py
├── scripts/                  # Batch job scripts
│   ├── batch_llmqa.sh       # Template batch script
│   └── examples/            # Example job scripts
│       └── os_jl_llmqa.sh   # Operating Systems course script
├── tests/                   # Test files
│   ├── test_critique_agent.py
│   ├── test_qa_generator.py
│   ├── test_rosie_llama.py
│   └── debug_qa_critique.py # Debug script for testing
├── chunks/                  # Text chunk data
│   └── opsys_chunks.csv    # Operating Systems course chunks
├── environment.yml          # Conda environment
├── requirements.txt         # Python dependencies
├── setup.py                # Package setup file
└── README.md               # Documentation
```

## License

Open Source (OSI Approved): [MIT License](LICENSE)
