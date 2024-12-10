"""Tests for the DatabricksModel class.

This module contains tests for the DatabricksModel functionality.
Run this file directly to execute the tests:
    python -m tests.test_databricks
"""

import unittest
from unittest.mock import patch, MagicMock
import os
from llmqa.models.databricks import DatabricksModel

class TestDatabricksModel(unittest.TestCase):
    """Test cases for DatabricksModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'DATABRICKS_URL': 'http://test-url',
            'DATABRICKS_KEY': 'test-key'
        }, clear=True)  # Clear all other env vars
        self.env_patcher.start()
        
        # Mock OpenAI client
        self.openai_patcher = patch('llmqa.models.databricks.OpenAI')
        self.mock_openai = self.openai_patcher.start()
        
        # Set up mock client
        self.mock_client = MagicMock()
        self.mock_openai.return_value = self.mock_client
        
        # Set up mock response
        mock_message = MagicMock()
        mock_message.content = "Test response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        # Create model instance
        self.model = DatabricksModel()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.env_patcher.stop()
        self.openai_patcher.stop()
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        self.assertEqual(self.model.model_name, "databricks-dbrx-instruct")
        self.assertIsInstance(self.model.prompt, list)
        self.assertEqual(len(self.model.prompt), 1)
        self.assertEqual(self.model.prompt[0]["role"], "system")
        
        # Check OpenAI client initialization
        self.mock_openai.assert_called_once_with(
            base_url='http://test-url',
            api_key='test-key'
        )
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        custom_prompt = "Custom system prompt"
        custom_url = "https://custom.databricks.com"
        model = DatabricksModel(
            model_name="databricks-dbrx-instruct",
            system_prompt=custom_prompt,
            base_url=custom_url
        )
        self.assertEqual(model.model_name, "databricks-dbrx-instruct")
        self.assertEqual(model.prompt[0]["content"], custom_prompt)
    
    def test_init_missing_env_vars(self):
        """Test initialization with missing environment variables."""
        # Stop current patcher and create new one with empty environment
        self.env_patcher.stop()
        empty_env_patcher = patch.dict('os.environ', {}, clear=True)
        empty_env_patcher.start()
        
        try:
            with self.assertRaises(ValueError) as context:
                DatabricksModel()
            self.assertIn("DATABRICKS_KEY and DATABRICKS_URL environment variables must be set", 
                         str(context.exception))
        finally:
            empty_env_patcher.stop()
            # Restore original environment for other tests
            self.env_patcher.start()
    
    def test_init_invalid_model(self):
        """Test initialization with invalid model name."""
        with self.assertRaises(ValueError) as context:
            DatabricksModel(model_name="invalid-model")
        self.assertIn("Model invalid-model not available", str(context.exception))
    
    def test_call_method(self):
        """Test the __call__ method."""
        response = self.model("Test message")
        
        # Verify response
        self.assertEqual(response, "Test response")
        
        # Verify API call
        self.mock_client.chat.completions.create.assert_called_once_with(
            model="databricks-dbrx-instruct",
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
    
    def test_available_models(self):
        """Test the available models list."""
        self.assertIn("databricks-dbrx-instruct", self.model.AVAILABLE_MODELS)
        self.assertIn("databricks-meta-llama-3-1-70b-instruct", self.model.AVAILABLE_MODELS)

def main():
    """Run the tests."""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main() 