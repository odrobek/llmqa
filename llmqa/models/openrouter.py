"""OpenRouter model implementation.

This module provides a wrapper around the OpenAI API for models hosted on OpenRouter.
"""

from openai import OpenAI
import os
from .base import BaseModel
import logging

logger = logging.getLogger('llmqa')


class OpenRouterModel(BaseModel):
    """OpenAI API wrapper for models hosted on OpenRouter. Only free models are supported
    which unfortunately have a rate limit of around 50 requests per day, which is very low.
    
    Attributes:
        client (OpenAI): The OpenAI client object
        prompt (list): The system prompt for the model
        model_name (str): The name of the model to use
    """
    
    AVAILABLE_MODELS = [
        "deepseek/deepseek-chat-v3-0324:free",
        "rekaai/reka-flash-3:free",
        "qwen/qwen2.5-vl-32b-instruct:free",
        "google/gemma-3-27b-it:free"
        # Add more models as they become available
    ]

    
    def __init__(self, 
                 model_name: str = "deepseek/deepseek-chat-v3-0324:free",
                 system_prompt: str = None,
                 base_url: str = "https://openrouter.ai/api/v1"):
        """Initialize the OpenRouter model.
        
        Args:
            model_name (str, optional): Name of the model to use. Defaults to "deepseek/deepseek-chat-v3-0324:free".
            system_prompt (str, optional): Custom system prompt. If None, uses default.
            base_url (str, optional): Custom base URL for the OpenRouter endpoint. 
                                    If None, uses environment variable.
        """
        token = os.getenv("OPENROUTER_KEY")
        base_url = base_url or os.getenv("OPENROUTER_URL")

        
        if not token or not base_url:
            logger.error("Missing required environment variables: OPENROUTER_KEY and/or OPENROUTER_URL")
            raise ValueError("OPENROUTER_KEY and OPENROUTER_URL environment variables must be set")
            
        if model_name not in self.AVAILABLE_MODELS:
            logger.error("Invalid model name: %s. Available models: %s", 
                        model_name, ", ".join(self.AVAILABLE_MODELS))
            raise ValueError(f"Model {model_name} not available. Choose from: {', '.join(self.AVAILABLE_MODELS)}")
        
        self.model_name = model_name
        logger.debug("Initializing OpenRouter model with base_url: %s, model: %s", base_url, model_name)
        self.client = OpenAI(
            base_url=base_url,
            api_key=token
        )

        default_prompt = """You are a helpful assistant that does everything asked of you."""

        self.prompt = [
            {"role": "system", "content": system_prompt or default_prompt},
        ]
        logger.debug("OpenRouter model initialized successfully")

    def __call__(self, message: str, max_tokens: int = 1024, temperature: float = 0.5, stream: bool = False) -> str:
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
        logger.debug("Sending API request to OpenRouter model: %s", self.model_name)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )
            logger.debug("Successfully received response from OpenRouter API")
            return completion.choices[0].message.content
        except Exception as e:
            logger.error("Error calling OpenRouter API: %s", str(e))
            raise 