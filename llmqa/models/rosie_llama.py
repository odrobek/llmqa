"""ROSIE Llama model implementation.

This module provides a wrapper around the OpenAI API for the Llama-3.1-70b model
hosted on ROSIE.
"""

from openai import OpenAI
import os
from .base import BaseModel

class ROSIELlama(BaseModel):
    """OpenAI API wrapper for the Llama-3.1-70b model hosted on ROSIE.
    
    Attributes:
        client (OpenAI): The OpenAI client object
        prompt (list): The system prompt for the model
    """
    
    def __init__(self, system_prompt: str = None):
        """Initialize the ROSIELlama model.
        
        Args:
            system_prompt (str, optional): Custom system prompt. If None, uses default.
        """
        url = os.getenv("ROSIE_URL")
        api_key = os.getenv("ROSIE_KEY")
        
        if not url or not api_key:
            raise ValueError("ROSIE_URL and ROSIE_KEY environment variables must be set")
        
        self.client = OpenAI(
            base_url=url,
            api_key=api_key
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
        """Process a message using the ROSIE Llama model.

        Args:
            message (str): The message to process

        Returns:
            str: The model's response
        """
        messages = [self.prompt[0], {"role": "user", "content": message}]
        completion = self.client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=messages,
            max_tokens=2048,
            temperature=0.9,
            stream=False
        )
        return completion.choices[0].message.content 