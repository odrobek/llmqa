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
        "databricks-dbrx-instruct",
        "databricks-meta-llama-3-1-70b-instruct",
        "databricks-mixtral-8x7b-instruct",
        "agents_poc_mosaic_sd_catalog-course_chunks-beta_model_v6"
        # Add more models as they become available
    ]

    
    def __init__(self, 
                 model_name: str = "databricks-dbrx-instruct",
                 system_prompt: str = None,
                 base_url: str = "https://dbc-e66ac7b6-520c.cloud.databricks.com/serving-endpoints"):
        """Initialize the Databricks model.
        
        Args:
            model_name (str, optional): Name of the model to use. Defaults to "databricks-dbrx-instruct".
            system_prompt (str, optional): Custom system prompt. If None, uses default.
            base_url (str, optional): Custom base URL for the Databricks endpoint. 
                                    If None, uses environment variable.
        """
        token = os.getenv("DATABRICKS_KEY")
        base_url = base_url or os.getenv("DATABRICKS_URL")

        
        if not token or not base_url:
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
            api_key=token
        )

        default_prompt = """You are a helpful assistant that does everything asked of you."""

        self.prompt = [
            {"role": "system", "content": system_prompt or default_prompt},
        ]
        logger.debug("Databricks model initialized successfully")

    def __call__(self, message: str) -> str:
        """Process a message using the Databricks model.

        Args:
            message (str): The message to process

        Returns:
            str: The model's response
        """
        messages = [self.prompt[0], {"role": "user", "content": message}]
        logger.debug("Sending API request to Databricks model: %s", self.model_name)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.5,
                stream=False
            )
            logger.debug("Successfully received response from Databricks API")
            return completion.choices[0].message.content
        except Exception as e:
            logger.error("Error calling Databricks API: %s", str(e))
            raise 