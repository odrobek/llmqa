"""Tests for the CritiqueAgent class.

This module contains tests for the CritiqueAgent functionality.
Run this file directly to execute the tests:
    python -m tests.test_critique_agent
"""

import unittest
from llmqa.models.critique_agent import CritiqueAgent
from llmqa.models.rosie_llama import ROSIELlama

class TestCritiqueAgent(unittest.TestCase):
    """Test cases for CritiqueAgent."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across all tests."""
        cls.model = ROSIELlama()
        cls.agent = CritiqueAgent(cls.model)
        
        # Test data
        cls.context = """
        The Hugging Face Hub is a platform where the machine learning community 
        collaborates on models, datasets, and applications. Users can easily 
        upload their models using git-based workflow and share them with the 
        community. The Hub supports many frameworks including PyTorch, TensorFlow, 
        and JAX.
        """
        
        cls.questions = {
            "good": "What is the Hugging Face Hub and what frameworks does it support?",
            "bad": "What is the price of the premium subscription?",
            "context_dependent": "What frameworks are mentioned in the text?",
            "irrelevant": "How do I make a sandwich?",
        }
    
    def test_parse_critique_valid_response(self):
        """Test parsing a valid critique response."""
        response = """
        Answer:::
        Evaluation: This is a test evaluation
        Total rating: 4.5
        """
        result = self.agent._parse_critique(response)
        
        self.assertIsInstance(result, dict)
        self.assertIn('rating', result)
        self.assertIn('evaluation', result)
        self.assertEqual(result['rating'], 4.5)
        self.assertEqual(result['evaluation'], "This is a test evaluation")
    
    def test_parse_critique_invalid_response(self):
        """Test parsing an invalid critique response."""
        invalid_responses = [
            "Invalid format",
            "Evaluation: Missing rating",
            "Total rating: 4.5\nMissing evaluation"
        ]
        
        for response in invalid_responses:
            with self.assertRaises(ValueError):
                self.agent._parse_critique(response)
    
    def test_groundedness_evaluation(self):
        """Test groundedness evaluation with good and bad questions."""
        # Test with a good question
        result = self.agent.evaluate_groundedness(
            self.questions["good"],
            self.context
        )
        self.assertIn('rating', result)
        self.assertIn('evaluation', result)
        self.assertGreaterEqual(result['rating'], 1)
        self.assertLessEqual(result['rating'], 5)
        
        # Test with a bad question
        result = self.agent.evaluate_groundedness(
            self.questions["bad"],
            self.context
        )
        self.assertIn('rating', result)
        self.assertIn('evaluation', result)
        self.assertGreaterEqual(result['rating'], 1)
        self.assertLessEqual(result['rating'], 5)
    
    def test_relevance_evaluation(self):
        """Test relevance evaluation."""
        result = self.agent.evaluate_relevance(self.questions["good"])
        self.assertIn('rating', result)
        self.assertIn('evaluation', result)
        self.assertGreaterEqual(result['rating'], 1)
        self.assertLessEqual(result['rating'], 5)
    
    def test_standalone_evaluation(self):
        """Test standalone evaluation with context-dependent and independent questions."""
        # Test with context-dependent question
        result = self.agent.evaluate_standalone(self.questions["context_dependent"])
        self.assertIn('rating', result)
        self.assertIn('evaluation', result)
        self.assertGreaterEqual(result['rating'], 1)
        self.assertLessEqual(result['rating'], 5)
        
        # Test with standalone question
        result = self.agent.evaluate_standalone(self.questions["good"])
        self.assertIn('rating', result)
        self.assertIn('evaluation', result)
        self.assertGreaterEqual(result['rating'], 1)
        self.assertLessEqual(result['rating'], 5)
    
    def test_evaluate_qa_pair_all_criteria(self):
        """Test evaluating a QA pair with all criteria."""
        result = self.agent.evaluate_qa_pair(
            self.questions["good"],
            self.context
        )
        
        self.assertIn('question', result)
        self.assertIn('critiques', result)
        self.assertIn('aggregate_score', result)
        
        critiques = result['critiques']
        self.assertIn('groundedness', critiques)
        self.assertIn('relevance', critiques)
        self.assertIn('standalone', critiques)
    
    def test_evaluate_qa_pair_specific_criteria(self):
        """Test evaluating a QA pair with specific criteria."""
        criteria = ['groundedness', 'relevance']
        result = self.agent.evaluate_qa_pair(
            self.questions["good"],
            self.context,
            criteria=criteria
        )
        
        self.assertIn('critiques', result)
        critiques = result['critiques']
        
        # Should only include specified criteria
        self.assertIn('groundedness', critiques)
        self.assertIn('relevance', critiques)
        self.assertNotIn('standalone', critiques)
    
    def test_evaluate_qa_pair_invalid_criteria(self):
        """Test evaluating a QA pair with invalid criteria."""
        with self.assertRaises(ValueError):
            self.agent.evaluate_qa_pair(
                self.questions["good"],
                self.context,
                criteria=['invalid_criterion']
            )

def main():
    """Run the tests with detailed output."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCritiqueAgent)
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    main() 