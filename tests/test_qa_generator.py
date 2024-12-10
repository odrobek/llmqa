"""Tests for the QAGenerator class.

This module contains tests for the QA generation functionality.
Run this file directly to execute the tests:
    python -m tests.test_qa_generator
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import tempfile
import os
from pathlib import Path
import pandas as pd

from llmqa.generators.qa_generator import QAGenerator, _try_parse_json
from llmqa.models.rosie_llama import ROSIELlama
from llmqa.models.critique_agent import CritiqueAgent

class TestQAGenerator(unittest.TestCase):
    """Test cases for QAGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the ROSIELlama model
        self.model_patcher = patch('llmqa.models.rosie_llama.ROSIELlama')
        self.mock_model_class = self.model_patcher.start()
        self.mock_model = MagicMock()
        self.mock_model_class.return_value = self.mock_model
        
        # Create generator instance
        self.model = ROSIELlama()
        self.generator = QAGenerator(self.model)
        
        # Sample valid QA pairs
        self.valid_qa_pairs = [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"},
            {"question": "Q3?", "answer": "A3"}
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.model_patcher.stop()
    
    def test_try_parse_json_valid(self):
        """Test JSON parsing with valid input."""
        valid_inputs = [
            # Clean JSON
            json.dumps(self.valid_qa_pairs),
            # JSON with whitespace
            f"\n  {json.dumps(self.valid_qa_pairs)}  \n",
            # JSON with single quotes
            str(self.valid_qa_pairs).replace('"', "'"),
            # Python list/dict notation
            str(self.valid_qa_pairs)
        ]
        
        for input_str in valid_inputs:
            result = _try_parse_json(input_str)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 3)
            for pair in result:
                self.assertIn('question', pair)
                self.assertIn('answer', pair)
    
    def test_try_parse_json_invalid(self):
        """Test JSON parsing with invalid input."""
        invalid_inputs = [
            "",  # Empty string
            "Not JSON at all",
            "{malformed: json}",
            "[1, 2, 3]",  # Valid JSON but wrong format
            "[{'incomplete': 'pair'}]",  # Missing required fields
            '[{"question": 1, "answer": "A1"}]',  # Wrong type for question
            '[{"question": "Q1", "answer": 2}]'  # Wrong type for answer
        ]
        
        for input_str in invalid_inputs:
            with self.assertRaises(ValueError):
                _try_parse_json(input_str)
    
    def test_generate_from_chunk_valid(self):
        """Test generating QA pairs from a valid chunk."""
        # Set up mock response with proper JSON
        mock_response = json.dumps(self.valid_qa_pairs)
        self.mock_model.return_value = mock_response
        
        # Test generation
        chunk = "Test chunk content"
        result = self.generator.generate_from_chunk(chunk)
        
        # Verify results
        self.assertIsInstance(result, tuple)
        all_pairs, filtered_pairs = result
        self.assertEqual(len(all_pairs), 3)
        self.assertEqual(len(filtered_pairs), 3)
        for pair in all_pairs:
            self.assertIn('question', pair)
            self.assertIn('answer', pair)
            self.assertIn('source_context', pair)
            self.assertEqual(pair['source_context'], chunk)
    
    def test_generate_from_chunk_invalid_response(self):
        """Test generating QA pairs with invalid model response."""
        invalid_responses = [
            "Not JSON",
            "[]",  # Empty array
            json.dumps([{"wrong": "format"}]),
            json.dumps([{"question": "Q1"}])  # Missing answer
        ]
        
        for response in invalid_responses:
            self.mock_model.return_value = response
            with self.assertRaises(ValueError):
                self.generator.generate_from_chunk("Test chunk")
    
    def test_generate_from_file(self):
        """Test generating QA pairs from a file."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            # Write CSV with proper column
            temp_file.write("processed_text\n")  # Header
            temp_file.write("Test content 1\n")
            temp_file.write("Test content 2\n")
            input_path = temp_file.name
        
        try:
            # Set up mock response
            mock_response = json.dumps(self.valid_qa_pairs)
            self.mock_model.return_value = mock_response
            
            # Create output path
            output_path = Path(input_path).with_suffix('.qa.csv')
            
            # Run generation
            self.generator.generate_from_file(input_path, output_path)
            
            # Verify output file exists and contains data
            self.assertTrue(output_path.exists())
            df = pd.read_csv(output_path)
            self.assertGreater(len(df), 0)
            
        finally:
            # Clean up temp files
            os.unlink(input_path)
            if output_path.exists():
                os.unlink(output_path)
    
    def test_generate_with_critique(self):
        """Test QA generation with critique agent."""
        # Set up mock response
        mock_response = json.dumps(self.valid_qa_pairs)
        self.mock_model.return_value = mock_response
        
        # Create generator with mock critique agent
        critique_agent = MagicMock()
        critique_agent.evaluate_qa_pair.return_value = {
            'critiques': ['Good question'],
            'aggregate_score': 4.0
        }
        
        generator = QAGenerator(
            self.model,
            critique_agent=critique_agent,
            min_critique_score=3.0
        )
        
        # Test generation
        all_pairs, filtered_pairs = generator.generate_from_chunk("Test chunk")
        
        # Verify results
        self.assertEqual(len(all_pairs), 3)
        self.assertEqual(len(filtered_pairs), 3)
        for pair in filtered_pairs:
            self.assertIn('critiques', pair)
            self.assertIn('aggregate_score', pair)
            self.assertGreaterEqual(pair['aggregate_score'], 3.0)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        with self.assertRaises(TypeError):
            self.generator.generate_from_chunk(None)
        
        with self.assertRaises(ValueError):
            self.generator.generate_from_chunk("")
        
        with self.assertRaises(FileNotFoundError):
            self.generator.generate_from_file(
                "nonexistent.csv",
                "output.json"
            )
        
        # Test with invalid CSV
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "invalid.csv"
            with open(input_path, 'w') as f:
                f.write("not,a,valid,csv")
            
            with self.assertRaises(ValueError):
                self.generator.generate_from_file(
                    input_path,
                    "output.json",
                    chunk_column='nonexistent_column'
                )

def main():
    """Run the tests."""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main() 