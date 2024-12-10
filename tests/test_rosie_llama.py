"""Tests for the ROSIELlama class.

This module contains tests for the ROSIELlama model functionality.
Run this file directly to execute the tests:
    python -m tests.test_rosie_llama
"""

import unittest
from unittest.mock import patch, MagicMock
import os
from llmqa.models.rosie_llama import ROSIELlama

class TestROSIELlama(unittest.TestCase):
    """Test cases for ROSIELlama."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'ROSIE_URL': 'http://test-url',
            'ROSIE_KEY': 'test-key'
        }, clear=True)  # Clear all other env vars
        self.env_patcher.start()
        
        # Mock OpenAI client
        self.openai_patcher = patch('llmqa.models.rosie_llama.OpenAI')
        self.mock_openai = self.openai_patcher.start()
        
        # Set up mock client
        self.mock_client = MagicMock()
        self.mock_openai.return_value = self.mock_client
        
        # Create model instance
        self.model = ROSIELlama()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.env_patcher.stop()
        self.openai_patcher.stop()
    
    def test_init_default_prompt(self):
        """Test initialization with default prompt."""
        self.assertEqual(
            self.model.prompt[0]['content'],
            "You are a helpful assistant that provides clear, accurate, and well-reasoned responses."
        )
        
        # Check OpenAI client initialization
        self.mock_openai.assert_called_once_with(
            base_url='http://test-url',
            api_key='test-key'
        )
    
    def test_init_custom_prompt(self):
        """Test initialization with custom prompt."""
        custom_prompt = "Custom system prompt"
        model = ROSIELlama(system_prompt=custom_prompt)
        self.assertEqual(model.prompt[0]['content'], custom_prompt)
    
    def test_init_missing_env_vars(self):
        """Test initialization with missing environment variables."""
        # Stop current patcher and create new one with empty environment
        self.env_patcher.stop()
        empty_env_patcher = patch.dict('os.environ', {}, clear=True)
        empty_env_patcher.start()
        
        try:
            with self.assertRaises(ValueError) as context:
                ROSIELlama()
            self.assertIn("ROSIE_URL and ROSIE_KEY environment variables must be set", str(context.exception))
        finally:
            empty_env_patcher.stop()
            # Restore original environment for other tests
            self.env_patcher.start()
    
    def test_call_method(self):
        """Test the __call__ method."""
        # Set up mock response
        mock_message = MagicMock()
        mock_message.content = "Test response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        # Test the call
        response = self.model("Test message")
        
        # Verify response
        self.assertEqual(response, "Test response")
        
        # Verify API call
        self.mock_client.chat.completions.create.assert_called_once_with(
            model="meta/llama-3.1-70b-instruct",
            messages=[
                self.model.prompt[0],
                {"role": "user", "content": "Test message"}
            ],
            max_tokens=2048,
            temperature=0.9,
            stream=False
        )
    
    def test_call_method_long_message(self):
        """Test the __call__ method with a long message."""
        long_message = "x" * 5000  # Create a very long message
        self.model(long_message)  # Should not raise any errors
        
        # Verify API call was made with the long message
        self.mock_client.chat.completions.create.assert_called_once()
        actual_message = self.mock_client.chat.completions.create.call_args[1]['messages'][1]['content']
        self.assertEqual(actual_message, long_message)
    
    def test_call_method_empty_message(self):
        """Test the __call__ method with an empty message."""
        response = self.model("")
        self.mock_client.chat.completions.create.assert_called_once()
        
    def test_call_method_special_characters(self):
        """Test the __call__ method with special characters."""
        special_chars = "!@#$%^&*()_+-=[]{}|;:'\",.<>?/\\"
        self.model(special_chars)
        self.mock_client.chat.completions.create.assert_called_once()
        actual_message = self.mock_client.chat.completions.create.call_args[1]['messages'][1]['content']
        self.assertEqual(actual_message, special_chars)

def main():
    """Run the tests."""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main() 