"""Critique Agent for evaluating question-answer pairs.

This module provides functionality for evaluating the quality of generated
question-answer pairs using various criteria.
"""

from typing import Dict, List, Any, Optional
from llmqa.models.base import BaseModel
import json
import logging

logger = logging.getLogger('llmqa')

# New default criterion prompt for evaluate_generated_answer
DEFAULT_GENERATED_ANSWER_CRITERION_PROMPT = """
Evaluate the generated answer based on its correctness and factual consistency with the provided context and its semantic similarity to the ground truth answer.
Provide your response strictly as valid JSON with no extra commentary. The JSON object must contain two keys:
1.  'evaluation': (string) Your detailed evaluation based on the criteria.
2.  'rating': (float) A numerical rating from 1.0 (worst) to 5.0 (best) reflecting the overall quality based on the criteria.

Context: {context}
Question: {question}
Ground Truth Answer: {ground_truth_answer}
Generated Answer: {generated_answer}

JSON:
"""

class CritiqueEvaluator:
    """Class for evaluating question-answer pairs using configurable criteria with LLM-as-a-judge."""
    
    def __init__(self, model: BaseModel):
        """Initialize the critique agent.
        
        Args:
            model (BaseModel): The LLM model to use for critiques
        """
        self.model = model
        logger.debug("Initialized CritiqueEvaluator.")

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
        criteria: List[Dict[str, Any]]
    ) -> Dict:
        """Evaluate a QA pair using the provided criteria.
        
        Args:
            question (str): The question to evaluate
            context (str): The context from which the question should be answerable
            answer (str): The answer to evaluate
            criteria (List[Dict[str, Any]]): List of criteria configurations to use for evaluation.
                 Each dict should have 'name', 'prompt_template', and optionally 'parameters'.
                
        Returns:
            Dict containing critiques for each criterion and aggregate score
        """
        # Use passed-in criteria directly
        logger.debug("Evaluating QA pair with %d provided criteria", len(criteria))
        
        # Initialize results
        critiques = {}
        
        # Evaluate each provided criterion
        for criterion in criteria: # Iterate over passed-in criteria
            criterion_name = criterion.get("name", "Unnamed Criterion") # Use a default name if missing
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
                    critiques[criterion_name] = self._parse_critique(response) # Use criterion_name
                    logger.debug("Successfully evaluated criterion: %s on attempt %d", criterion_name, attempt + 1)
                    # Break the retry loop if successful
                    break 
                    
                except Exception as e:
                    logger.warning(
                        "Attempt %d/%d failed for criterion %s: %s",
                        attempt + 1, max_retries, criterion_name, str(e)
                    )
                    if attempt + 1 == max_retries:
                        logger.error(
                            "All %d attempts failed for criterion %s. Assigning error critique.", 
                            max_retries, criterion_name
                        )
                        critiques[criterion_name] = {
                            "evaluation": f"Error after {max_retries} attempts: {str(e)}",
                            "rating": 1.0  # Default to lowest rating on error
                        }
        
        # Calculate aggregate score from all critiques
        ratings = [c["rating"] for c in critiques.values() if isinstance(c, dict) and "rating" in c]
        aggregate_score = sum(ratings) / len(ratings) if ratings else None
        
        return {
            "critiques": critiques,
            "aggregate_score": aggregate_score
        } 

    def evaluate_generated_answer(
        self,
        question: str,
        context: str,
        generated_answer: str,
        ground_truth_answer: str,
        criteria: Optional[List[Dict[str, Any]]] = None
    ) -> Dict:
        """Evaluate a generated answer against a ground truth answer.

        If criteria are provided, evaluates based on each criterion.
        If no criteria are provided, uses a default evaluation focusing on correctness,
        factual consistency with context, and semantic similarity to the ground truth.

        Args:
            question (str): The original question.
            context (str): The context relevant to the question.
            generated_answer (str): The answer generated by a model (without context).
            ground_truth_answer (str): The correct answer derived from the context.
            criteria (Optional[List[Dict[str, Any]]]): Optional list of criteria configurations.
                 Each dict should have 'name', 'prompt_template', and optionally 'parameters'.

        Returns:
            Dict containing critiques and aggregate score.
        """
        critiques = {}
        aggregate_score = None

        if criteria:
            # Evaluate using provided criteria list
            logger.debug("Evaluating generated answer with %d provided criteria", len(criteria))
            for criterion in criteria:
                criterion_name = criterion.get("name", "Unnamed Criterion")
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        params = {
                            "question": question,
                            "context": context,
                            "generated_answer": generated_answer,
                            "ground_truth_answer": ground_truth_answer
                        }
                        required_params = criterion.get("parameters", [])
                        formatted_params = {k: v for k, v in params.items() if k in required_params}
                        prompt = criterion["prompt_template"].format(**formatted_params)
                        
                        logger.debug("Sending request to evaluator model for criterion '%s' (Attempt %d/%d)", criterion_name, attempt + 1, max_retries)
                        response = self.model(prompt)
                        critiques[criterion_name] = self._parse_critique(response)
                        logger.debug("Successfully evaluated criterion: %s on attempt %d", criterion_name, attempt + 1)
                        break
                    except Exception as e:
                        logger.warning(
                            "Attempt %d/%d failed for criterion %s: %s",
                            attempt + 1, max_retries, criterion_name, str(e)
                        )
                        if attempt + 1 == max_retries:
                            logger.error(
                                "All %d attempts failed for criterion %s. Assigning error critique.", 
                                max_retries, criterion_name
                            )
                            critiques[criterion_name] = {
                                "evaluation": f"Error after {max_retries} attempts: {str(e)}",
                                "rating": 1.0
                            }
            # Calculate aggregate score from all provided criteria critiques
            ratings = [c["rating"] for c in critiques.values() if isinstance(c, dict) and "rating" in c]
            aggregate_score = sum(ratings) / len(ratings) if ratings else None
            logger.debug("Completed evaluation using provided criteria. Aggregate score: %s", aggregate_score)

        else:
            # Evaluate using the default single criterion
            logger.debug("Evaluating generated answer against ground truth using default criterion.")
            critique_result = {}
            critique_name = "Generated Answer Correctness vs Ground Truth"
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    prompt = DEFAULT_GENERATED_ANSWER_CRITERION_PROMPT.format(
                        question=question,
                        context=context,
                        generated_answer=generated_answer,
                        ground_truth_answer=ground_truth_answer
                    )
                    
                    logger.debug("Sending request to evaluator model (Attempt %d/%d)", attempt + 1, max_retries)
                    response = self.model(prompt)
                    critique_result = self._parse_critique(response)
                    logger.debug("Successfully evaluated generated answer on attempt %d", attempt + 1)
                    break # Break retry loop if successful

                except Exception as e:
                    logger.warning(
                        "Attempt %d/%d failed for default generated answer evaluation: %s",
                        attempt + 1, max_retries, str(e)
                    )
                    if attempt + 1 == max_retries:
                        logger.error(
                            "All %d attempts failed for default generated answer evaluation. Assigning error critique.", 
                            max_retries
                        )
                        critique_result = {
                            "evaluation": f"Error after {max_retries} attempts: {str(e)}",
                            "rating": 1.0  # Default to lowest rating on error
                        }

            critiques = {critique_name: critique_result}
            # The aggregate score is just the rating from the single critique
            aggregate_score = critique_result.get("rating") if isinstance(critique_result, dict) else None
            logger.debug("Completed evaluation using default criterion. Aggregate score: %s", aggregate_score)

        return {
            "critiques": critiques,
            "aggregate_score": aggregate_score
        } 