"""ROSIE Llama model implementation.

This module provides a wrapper around the OpenAI API for the Llama-3.1-70b model
hosted on ROSIE.
"""

from openai import OpenAI
import os
from .base import BaseModel
import logging

logger = logging.getLogger('llmqa')

class ROSIELlama(BaseModel):
    """OpenAI API wrapper for the Llama-3.1-70b model hosted on ROSIE.


    Attributes:
        client (OpenAI): The OpenAI client object
        prompt (list): The system prompt for the model
    """
    
    def __init__(self, model_name: str = None, base_url: str = None, system_prompt: str = None):
        """Initialize the ROSIELlama model.
        
        Args:
            system_prompt (str, optional): Custom system prompt. If None, uses default.
        """
        url = os.getenv("ROSIE_URL")
        api_key = os.getenv("ROSIE_KEY")

        self.model_name = model_name
        
        if not url or not api_key:
            logger.error("Missing required environment variables: ROSIE_URL and/or ROSIE_KEY")
            raise ValueError("ROSIE_URL and ROSIE_KEY environment variables must be set")
        
        logger.debug("Initializing ROSIE Llama model with base_url: %s", url)
        self.client = OpenAI(
            base_url=url,
            api_key=api_key
        )

        default_prompt = """You are a helpful assistant that does everything asked of you."""

        self.prompt = [
            {"role": "system", "content": system_prompt or default_prompt},
        ]
        logger.debug("ROSIE Llama model initialized successfully")

    def __call__(self, message: str, max_tokens: int = 2048, temperature: float = 0.9, stream: bool = False) -> str:
        """Process a message using the ROSIE Llama model.

        Args:
            message (str): The message to process
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 2048.
            temperature (float, optional): The temperature to use for the model. Defaults to 0.9.
            stream (bool, optional): Whether to stream the response. Defaults to False. Recommended to not change.

        Returns:
            str: The model's response
        """
        messages = [self.prompt[0], {"role": "user", "content": message}]
        logger.debug("Sending API request to ROSIE Llama model: %s", self.model_name)

        try:
            completion = self.client.chat.completions.create(
                model="meta/llama-3.3-70b-instruct",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )
            logger.debug("Successfully received response from ROSIE Llama API")
            return completion.choices[0].message.content 
        except Exception as e:
            logger.error("Error calling ROSIE Llama API: %s", str(e))
            raise e
