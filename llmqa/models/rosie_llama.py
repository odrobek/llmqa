"""ROSIE Llama model implementation.

This module connects to ROSIE with paramiko and an ssh connection and 
provides a wrapper to query Llama models hosted on ROSIE.
"""

from openai import OpenAI
import paramiko
import os
from .base import BaseModel
import logging

logger = logging.getLogger('llmqa')

class ROSIELlama(BaseModel):
    """SSH connection wrapper for the Llama models hosted on ROSIE.


    Attributes:
        client (OpenAI): The OpenAI client object
        prompt (list): The system prompt for the model
    """

    AVAILABLE_MODELS = [
        "meta/llama-3.3-70b-instruct",
        "meta/llama-3.2-90b-vision-instruct"
    ]
    
    def __init__(self, model_name: str = None, 
                 base_url: str = None,
                 ssh_username: str = None,
                 ssh_password: str = None,
                 ssh_hostname: str = None,
                 system_prompt: str = None):
        """Initialize the ROSIELlama model.
        
        Args:
            system_prompt (str, optional): Custom system prompt. If None, uses default.
        """
        if not ssh_username or not ssh_password or not ssh_hostname:
            logger.error("ssh_username, ssh_password, and ssh_hostname must be set")
            raise ValueError("ssh_username, ssh_password, and ssh_hostname must be set")
        url = base_url or os.getenv("ROSIE_URL")
        api_key = os.getenv("ROSIE_KEY")

        self.model_name = model_name

        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=ssh_hostname, port=22, username=ssh_username, password=ssh_password)

        # Different ports for the two models
        if model_name == "meta/llama-3.3-70b-instruct":
            url = url.replace("8001", "8000")
        elif model_name == "meta/llama-3.2-90b-vision-instruct":
            url = url.replace("8000", "8001")

        if model_name not in self.AVAILABLE_MODELS:
            logger.error("Invalid model name: %s. Available models: %s", 
                        model_name, ", ".join(self.AVAILABLE_MODELS))
            raise ValueError(f"Model {model_name} not available. Choose from: {', '.join(self.AVAILABLE_MODELS)}")
        
        # if not url or not api_key:
        #     logger.error("Missing required environment variables: ROSIE_URL and/or ROSIE_KEY")
        #     raise ValueError("ROSIE_URL and ROSIE_KEY environment variables must be set")
        
        # logger.debug("Initializing ROSIE Llama model with base_url: %s", url)
        # self.client = OpenAI(
        #     base_url=url,
        #     api_key=api_key
        # )

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
            stdin, stdout, stderr = self.ssh.exec_command(f'cd rosie_llama && source $HOME/.local/bin/env && uv run try_rosie_llm.py --message "{message}" --model "{self.model_name}"')
            return stdout.read().decode()
            # completion = self.client.chat.completions.create(
            #     model="meta/llama-3.3-70b-instruct",
            #     messages=messages,
            #     max_tokens=max_tokens,
            #     temperature=temperature,
            #     stream=stream
            # )
            # logger.debug("Successfully received response from ROSIE Llama API")
            # return completion.choices[0].message.content 
        except Exception as e:
            logger.error("Error calling ROSIE Llama API: %s", str(e))
            raise e
