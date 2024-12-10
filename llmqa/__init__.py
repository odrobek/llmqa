"""LLMQA - Large Language Model Question and Answer Generation

This package provides functionality for generating question-answer pairs from text
using large language models.
"""

from .models.rosie_llama import ROSIELlama
from .models.critique_agent import CritiqueAgent
from .models.databricks import DatabricksModel
from .generators.qa_generator import QAGenerator

__version__ = "0.1.0"
__all__ = ["ROSIELlama", "DatabricksModel", "CritiqueAgent", "QAGenerator"]