# llmqa/core/llm_service.py

from typing import List, Dict, Any, Optional
import ast
import threading
from llmqa.models.base import BaseModel
from llmqa.evaluators.critique import CritiqueEvaluator
import logging

logger = logging.getLogger('llmqa')

class LLMService:
    def __init__(
        self, 
        model: BaseModel,
        evaluator: Optional[CritiqueEvaluator] = None,
    ):
        """
        Unified interface for LLM calls.
        
        Args:
            model (BaseModel): An instance of a model to be used for generating responses.
            evaluator (CritiqueEvaluator, optional): An evaluator for whatever specified criteria are used.
        """
        self.model = model
        self.evaluator = evaluator
        logger.debug("Initialized LLMService with model: %s, evaluator: %s", 
                    model.__class__.__name__, 
                    evaluator.__class__.__name__ if evaluator else None)


    def generate_qa(self, text_chunk: str, with_critique: bool = False, criteria: Optional[List[str]] = None, cancel_event: Optional[threading.Event] = None) -> List[Dict[str, Any]]:
        """
        Generate question-answer pairs from a text chunk.
        
        Args:
            text_chunk (str): The input text.
            with_critique (bool): Whether to run evaluation/critique on the generated pairs.
            criteria (List[str], optional): Specific evaluation criteria.
            
        Returns:
            A list of QA pairs, optionally augmented with critique-evaluation data.
        """
        logger.debug("Starting QA generation for text chunk (length: %d)", len(text_chunk))
        if cancel_event and cancel_event.is_set():
            logger.debug("QA generation cancelled before starting")
            return []

        GENQA_PROMPT = """
        You are a helpful assistant that generates ONE AND ONLY ONE question-answer pair from the given text chunk.
        The text chunk is a section of a larger text, and the question-answer pair should be relevant to the text chunk.
        The question-answer pair should be in the form of a list of dictionaries, where each dictionary contains a question and an answer.
        The question should have substance to it, be relevant to the text chunk, and should NOT be answerable with a simple word or phrase.

        The question should be one that a student might ask a professor in a college course. The answer should be able to be found in the given
        text chunk, and ideally follow a similar grammar and style to the text chunk. The question and answer pair will be evaluated
        based on arbritrary criteria (such as but not limited to: groundedness, relevance, accuracy) after generation. 

        Provide your answers strictly as valid JSON with no extra commentary following the JSON: at the bottom of this prompt.
        Your response should start with [ and end with ], and the ONE question and answer pair should have the following format:
        'question': <question>, 'answer': <answer> as a dictionary.

        Here is the text chunk:
        {text_chunk}

        JSON:"""

        # Call the model and parse its response
        prompt = GENQA_PROMPT.format(text_chunk=text_chunk)
        logger.debug("Sending API request to model for QA generation")
        response = self.model(prompt)
        
        if cancel_event and cancel_event.is_set():
            logger.debug("QA generation cancelled after model response")
            return []
        
        try:
            # Attempt to clean the response, e.g., remove backticks if present
            cleaned_response = response.strip().strip("```json").strip("```")
            # Parse the JSON
            qa_pairs = ast.literal_eval(cleaned_response)
            logger.debug("Successfully parsed %d QA pairs from model response", len(qa_pairs))
        except Exception as e:
            logger.error("Failed to parse model response: %s", str(e))
            raise ValueError(f"Failed to parse model response: {e}")
        
        if not isinstance(qa_pairs, list):
            logger.error("Model response was not a list of QA pairs")
            raise ValueError("Model response must be a list of QA pairs.")
        
        # If evaluation is requested, use the evaluator.
        if with_critique and self.evaluator:
            logger.debug("Starting critique evaluation for %d QA pair", len(qa_pairs))
            for i, pair in enumerate(qa_pairs):
                if cancel_event and cancel_event.is_set():
                    logger.debug("QA generation cancelled during critique evaluation")
                    return qa_pairs[:i]
    
                logger.debug("Evaluating QA pair %d/%d", i+1, len(qa_pairs))

                critiques = self.evaluator.evaluate_qa_pair(
                    question=pair.get('question'),
                    context=text_chunk,
                    answer=pair.get('answer'),
                )

                pair['critiques'] = critiques['critiques']
                pair['aggregate_score'] = critiques['aggregate_score']
                
            logger.debug("Completed critique evaluation for all QA pairs")
        return qa_pairs