"""Critique Agent for evaluating question-answer pairs.

This module provides functionality for evaluating the quality of generated
question-answer pairs using various criteria.
"""

from typing import Dict, List, Any
from llmqa.models.base import BaseModel
import json
import logging

logger = logging.getLogger('llmqa')

class CritiqueEvaluator:
    """Class for evaluating question-answer pairs using configurable criteria with LLM-as-a-judge."""
    
    def __init__(self, model: BaseModel, criteria: List[Dict[str, Any]] = None):
        """Initialize the critique agent.
        
        Args:
            model (BaseModel): The LLM model to use for critiques
            criteria (List[Dict[str, Any]], optional): List of criteria configurations
        """
        self.model = model
        self.criteria = criteria or []
        logger.debug("Initialized CritiqueEvaluator with %d criteria", len(self.criteria))

    def _parse_critique(self, response: str) -> dict:
        """Parse the critique response from the model.
        
        Args:
            response (str): The raw response from the model
            
        Returns:
            dict: Parsed critique containing evaluation and rating
        """
        try:
            # Clean the response
            cleaned_response = response.replace("json", "").strip().strip("`")
            # Parse the JSON
            parsed = json.loads(cleaned_response)
            # Validate required keys
            if "evaluation" not in parsed or "rating" not in parsed:
                raise ValueError("Missing required keys in JSON response")
            parsed['rating'] = float(parsed['rating'])
            return parsed
        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON response: %s", str(e))
            raise ValueError(f"Failed to decode JSON: {e}\nResponse: {response}")
    
    def evaluate_qa_pair(
        self, 
        question: str, 
        context: str,
        answer: str,
    ) -> Dict:
        """Evaluate a QA pair using all enabled criteria.
        
        Args:
            question (str): The question to evaluate
            context (str): The context from which the question should be answerable
            answer (str): The answer to evaluate
                
        Returns:
            Dict containing critiques for each criterion and aggregate score
        """
        # Get enabled criteria
        enabled_criteria = [c for c in self.criteria if c.get("enabled", False)]
        logger.debug("Evaluating QA pair with %d enabled criteria", len(enabled_criteria))
        
        # Initialize results
        critiques = {}
        
        # Evaluate each enabled criterion
        for criterion in enabled_criteria:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Prepare parameters based on what the criterion needs
                    params = {
                        "question": question,
                        "context": context,
                        "answer": answer
                    }
                    
                    # Only include parameters that the criterion expects
                    required_params = criterion.get("parameters", [])
                    formatted_params = {k: v for k, v in params.items() if k in required_params}
                    
                    # Format the prompt template with the required parameters
                    prompt = criterion["prompt_template"].format(**formatted_params)
                    
                    # Get and parse the model's response
                    response = self.model(prompt)
                    critiques[criterion["name"]] = self._parse_critique(response)
                    logger.debug("Successfully evaluated criterion: %s on attempt %d", criterion["name"], attempt + 1)
                    # Break the retry loop if successful
                    break 
                    
                except Exception as e:
                    logger.warning(
                        "Attempt %d/%d failed for criterion %s: %s",
                        attempt + 1, max_retries, criterion["name"], str(e)
                    )
                    if attempt + 1 == max_retries:
                        logger.error(
                            "All %d attempts failed for criterion %s. Assigning error critique.", 
                            max_retries, criterion["name"]
                        )
                        critiques[criterion["name"]] = {
                            "evaluation": f"Error after {max_retries} attempts: {str(e)}",
                            "rating": 1.0  # Default to lowest rating on error
                        }
        
        # Calculate aggregate score from all critiques
        ratings = [c["rating"] for c in critiques.values()]
        aggregate_score = sum(ratings) / len(ratings) if ratings else None
        
        return {
            "critiques": critiques,
            "aggregate_score": aggregate_score
        } 