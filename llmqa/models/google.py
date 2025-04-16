"""Google model implementation.

This module provides a wrapper around the OpenAI API for models hosted on Google Cloud.
"""

from google import genai
from google.genai import types
import os
from .base import BaseModel
import logging

logger = logging.getLogger()

class GoogleModel(BaseModel):
    """OpenAI API wrapper for models hosted on Google Cloud.
    
    Attributes:
        client (genai.Client): The Google client object
        prompt (list): The system prompt for the model
        model_name (str): The name of the model to use
    """
    
    AVAILABLE_MODELS = [
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
        "gemini-2.0-flash-lite"
    ]
    
    def __init__(self, 
                 model_name: str = "gemini-1.5-flash",
                 system_prompt: str = None,
                 base_url: str = "not_used"):
        """Initialize the Google model.
        
        Args:
            model_name (str, optional): Name of the model to use. Defaults to "gemini-1.5-flash".
            system_prompt (str, optional): Custom system prompt. If None, uses default.
            base_url (str, optional): Custom base URL for the Google endpoint. 
                                    If None, uses environment variable.
        """
        token = os.getenv("GOOGLE_KEY")
        base_url = base_url or os.getenv("GOOGLE_URL")
        
        if not token or not base_url:
            logger.error("Missing required environment variables: GOOGLE_KEY and/or GOOGLE_URL")
            raise ValueError("GOOGLE_KEY and GOOGLE_URL environment variables must be set")
            
        if model_name not in self.AVAILABLE_MODELS:
            logger.error("Invalid model name: %s. Available models: %s", 
                        model_name, ", ".join(self.AVAILABLE_MODELS))
            raise ValueError(f"Model {model_name} not available. Choose from: {', '.join(self.AVAILABLE_MODELS)}")
        
        self.model_name = model_name
        logger.debug("Initializing Google model with base_url: %s, model: %s", base_url, model_name)
        self.client = genai.Client(api_key=token)

        default_prompt = """You are a helpful assistant that does everything asked of you."""

        self.prompt = [
            {"role": "system", "content": system_prompt or default_prompt},
        ]
        logger.debug("Google model initialized successfully")

    def __call__(self, message: str, max_tokens: int = 1024, temperature: float = 0.5, stream: bool = False) -> str:
        """Process a message using the Google model.

        Args:
            message (str): The message to process
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
            temperature (float, optional): The temperature to use for the model. Defaults to 0.5.
            stream (bool, optional): Whether to stream the response. Defaults to False. Recommended to not change.


        Returns:
            str: The model's response
        """
        messages = [self.prompt[0], {"role": "user", "content": message}]
        logger.debug("Sending API request to Google model: %s", self.model_name)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[message],
                config=types.GenerateContentConfig(
                    system_instruction=self.prompt[0]["content"],
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            logger.debug("Successfully received response from Google API")
            return response.text
        except Exception as e:
            logger.error("Error calling Google API: %s", str(e))
            raise 