"""Databricks model implementation.

This module provides a wrapper around the OpenAI API for models hosted on Databricks.
"""

from openai import OpenAI
import os
from .base import BaseModel
import logging

logger = logging.getLogger('llmqa')


class DatabricksModel(BaseModel):
    """OpenAI API wrapper for models hosted on Databricks.
    
    Attributes:
        client (OpenAI): The OpenAI client object
        prompt (list): The system prompt for the model
        model_name (str): The name of the model to use
    """
    
    AVAILABLE_MODELS = [
        "databricks-meta-llama-3-3-70b-instruct",
        "databricks-meta-llama-3-1-8b-instruct",
        "databricks-claude-3-7-sonnet",
        "databricks-llama-4-maverick",
        "agents_poc_mosaic_sd_catalog-course_chunks-beta_model_v7"
    ]

    
    def __init__(self, 
                 model_name: str = "databricks-meta-llama-3-3-70b-instruct",
                 system_prompt: str = None,
                 api_key: str = None,
                 base_url: str = None):
        """Initialize the Databricks model.
        
        Args:
            model_name (str, optional): Name of the model to use. Defaults to "databricks-meta-llama-3-3-70b-instruct".
            system_prompt (str, optional): Custom system prompt. If None, uses default.
            base_url (str, optional): Custom base URL for the Databricks endpoint. 
                                    If None, uses environment variable.
        """
        api_key = api_key or os.getenv("DATABRICKS_KEY")
        base_url = base_url or os.getenv("DATABRICKS_URL")

        
        if not api_key or not base_url:
            logger.error("Missing required environment variables: DATABRICKS_KEY and/or DATABRICKS_URL")
            raise ValueError("DATABRICKS_KEY and DATABRICKS_URL environment variables must be set")
            
        if model_name not in self.AVAILABLE_MODELS:
            logger.error("Invalid model name: %s. Available models: %s", 
                        model_name, ", ".join(self.AVAILABLE_MODELS))
            raise ValueError(f"Model {model_name} not available. Choose from: {', '.join(self.AVAILABLE_MODELS)}")
        
        self.model_name = model_name
        logger.debug("Initializing Databricks model with base_url: %s, model: %s", base_url, model_name)
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        default_prompt = """You are a helpful assistant that does everything asked of you."""

        self.prompt = [
            {"role": "system", "content": system_prompt or default_prompt},
        ]
        logger.debug("Databricks model initialized successfully")

    def __call__(self, message: str, max_tokens: int = 1024, temperature: float = 0.5, stream: bool = False, course_uuid: str = None) -> str:
        """Process a message using the Databricks model.

        Args:
            message (str): The message to process
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
            temperature (float, optional): The temperature to use for the model. Defaults to 0.5.
            stream (bool, optional): Whether to stream the response. Defaults to False. Recommended to not change.

        Returns:
            str: The model's response
        """
        messages = [self.prompt[0], {"role": "user", "content": message}]
        logger.debug("Sending API request to Databricks model: %s", self.model_name)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                extra_body={"course_uuid": course_uuid, "deprecated_column0": "na"}
            )
            logger.debug("Response from Databricks API: %s", completion)
            logger.debug("Successfully received response from Databricks API")
            return completion.choices[0].message.content
        except Exception as e:
            logger.error("Error calling Databricks API: %s", str(e))
            raise 