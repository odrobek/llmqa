"""Databricks model implementation.

This module provides a wrapper around the OpenAI API for models hosted on Databricks.
"""

from openai import OpenAI
import os
from .base import BaseModel

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
        # Add more models as they become available
    ]
    
    def __init__(self, 
                 model_name: str = "databricks-dbrx-instruct",
                 system_prompt: str = None,
                 base_url: str = None):
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
            raise ValueError("DATABRICKS_KEY and DATABRICKS_URL environment variables must be set")
            
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not available. Choose from: {', '.join(self.AVAILABLE_MODELS)}")
        
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=token
        )

        default_prompt = """You are a helpful assistant that does everything asked of you. Each prompt given to you will be a chunk of information from a knowledge database. 
            You are helping to create Question and Answer pairs from each individual chunk. Create 3 questions of substance for each chunk and ONLY 3. The question and answer must be found in the chunk. 
            Make sure that questions have the correct context, NEVER ASK A QUESTION without the proper context to answer it. Do not assume that the person you are asking the question of can view the chunk.
            For example, do not say "from the given example" without providing the example. 
            Your output will be json and ONLY json (with NO COMMENTS AT ALL), with each object being a question answer pair. Your response should start with [ and end with ]. 
            Make sure that the questions require more than one sentence to answer. Your response should NEVER be more than 2048 tokens."""

        self.prompt = [
            {"role": "system", "content": system_prompt or default_prompt},
        ]

    def __call__(self, message: str) -> str:
        """Process a message using the Databricks model.

        Args:
            message (str): The message to process

        Returns:
            str: The model's response
        """
        messages = [self.prompt[0], {"role": "user", "content": message}]
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=2048,
            temperature=0.9,
            stream=False
        )
        return completion.choices[0].message.content 