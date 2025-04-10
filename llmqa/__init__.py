"""LLMQA - Large Language Model Question and Answer Generation

This package provides functionality for generating question-answer pairs from text
using large language models.
"""

from .models.rosie_llama import ROSIELlama
from .models.databricks import DatabricksModel
from .models.google import GoogleModel
from .models.openrouter import OpenRouterModel
from .core.llm_service import LLMService
from .evaluators import CritiqueEvaluator


__version__ = "0.1.0"
__all__ = ["ROSIELlama", "DatabricksModel", "GoogleModel", "OpenRouterModel", "LLMService", "CritiqueEvaluator"]