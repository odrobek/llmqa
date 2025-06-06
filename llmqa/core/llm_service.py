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


    def generate_qa(self, text_chunk: str, with_critique: bool = False, criteria: Optional[List[Dict[str, Any]]] = None, 
                    gen_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate question-answer pairs from a text chunk.
        
        Args:
            text_chunk (str): The input text.
            with_critique (bool): Whether to run evaluation/critique on the generated pairs.
            criteria (Optional[List[Dict[str, Any]]]): List of criteria configurations to use for evaluation.
                 Required if `with_critique` is True. Each dict should have 'name',
                 'prompt_template', and optionally 'parameters'.
            
        Returns:
            A list of QA pairs, optionally augmented with critique-evaluation data.
        """
        logger.debug("Starting QA generation for text chunk (length: %d)", len(text_chunk))

        GENQA_PROMPT = gen_prompt if gen_prompt else """
        You are a helpful assistant that generates ONE AND ONLY ONE question-answer pair from the given text chunk.
        Your purpose is to help generate an individualized validation dataset for a given educational course. This will be used to evaluate the specificity of
        models on the course content, and the models evaluated will range from RAG to general-purpose LLMs.

        The text chunk is a section of a larger text which might be class notes, transcripts, a textbook, etc. The question-answer pair you generate MUST be relevant to the text chunk.
        The question MUST have substance to it, be relevant to the text chunk, and should NOT be answerable with a simple word or phrase.

        The question should be one that a student might ask a professor in a college course. The answer should be able to be found in the given
        text chunk, and follow a similar grammar and style to the text chunk. The question and answer pair will be evaluated
        based on arbritrary criteria (such as but not limited to: groundedness, relevance, accuracy) after generation. 
        Questions should be specific to course content, and should not be contingent on knowing about some external context (for example, do NOT mention "from the passage" or "from the text").
        Additionally, questions must not be about individual specific assignments, projects, or dates/locations.


        Provide your output strictly as valid JSON with no extra commentary following the JSON: at the bottom of this prompt.
        The valid JSON should have two keys- "question" and "answer". These keys should be a dictionary (the valid JSON).

        Now here is the text chunk:
        {text_chunk}

        JSON:"""

        # Call the model and parse its response
        prompt = GENQA_PROMPT.format(text_chunk=text_chunk)
        logger.debug("Sending API request to model for QA generation")
        response = self.model(prompt)
        
        try:
            # Attempt to clean the response, remove backticks if present
            cleaned_response = response.strip().strip("```json").strip("```")
            # Parse the JSON
            qa_pairs = ast.literal_eval(cleaned_response)
            qa_pairs = [qa_pairs]
            logger.debug("Successfully parsed %d QA pairs from model response", len(qa_pairs))
        except Exception as e:
            logger.error("Failed to parse model response: %s", str(e))
            raise ValueError(f"Failed to parse model response: {e}")
        
        if not isinstance(qa_pairs, list):
            logger.error("Model response was not a list of QA pairs")
            raise ValueError("Model response must be a list of QA pairs.")
        
        # If evaluation is requested, use the evaluator.
        if with_critique and self.evaluator:
            # Check if criteria were provided when critique is requested
            if not criteria:
                logger.error("Critique requested but no criteria provided.")
                raise ValueError("Criteria must be provided when 'with_critique' is True.")
            
            logger.debug("Starting critique evaluation for %d QA pair with %d criteria", len(qa_pairs), len(criteria))
            for i, pair in enumerate(qa_pairs):
                logger.debug("Evaluating QA pair %d/%d", i+1, len(qa_pairs))

                # Pass the criteria list to the evaluator
                critiques = self.evaluator.evaluate_qa_pair(
                    question=pair.get('question'),
                    context=text_chunk,
                    answer=pair.get('answer'),
                    criteria=criteria # Pass criteria here
                )

                pair['critiques'] = critiques['critiques']
                pair['aggregate_score'] = critiques['aggregate_score']
                
            logger.debug("Completed critique evaluation for all QA pairs")
        return qa_pairs

    def generate_and_evaluate_answer(
        self,
        question: str,
        context: str,
        ground_truth_answer: str,
        criteria: Optional[List[Dict[str, Any]]] = None,
        course_uuid: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates an answer using the model based only on the question, then evaluates it
        against the ground truth answer and context using the evaluator.

        Args:
            question (str): The question to ask the generation model.
            context (str): The context used for evaluation.
            ground_truth_answer (str): The reference answer for evaluation.
            criteria (Optional[List[Dict[str, Any]]]): Optional list of criteria configurations
                 for the evaluator. If None, the evaluator's default logic is used.

        Returns:
            A dictionary containing the question, generated answer, ground truth,
            critiques, and aggregate score.
        """
        logger.debug("Starting answer generation for question: '%s'", question)

        # 1. Generate answer using the model with only the question
        # Per user request, the prompt is just the question itself
        course_uuid = course_uuid if course_uuid else None
        generation_prompt = question 
        logger.debug("Sending request to generation model")
        if course_uuid:
            generated_answer = self.model(generation_prompt, course_uuid=course_uuid)
        else:
            generated_answer = self.model(generation_prompt)
        logger.debug("Received generated answer (length: %d)", len(generated_answer))

        # 2. Evaluate the generated answer if an evaluator is available
        critiques = {}
        aggregate_score = None

        if self.evaluator:
            logger.debug("Starting evaluation of generated answer using evaluator")
            try:
                # Pass criteria to the evaluator method
                evaluation_results = self.evaluator.evaluate_generated_answer(
                    question=question,
                    context=context,
                    generated_answer=generated_answer,
                    ground_truth_answer=ground_truth_answer,
                    criteria=criteria # Pass criteria here
                )
                critiques = evaluation_results.get('critiques', {})
                aggregate_score = evaluation_results.get('aggregate_score')
                logger.debug("Completed evaluation of generated answer")
            except Exception as e:
                logger.error("Failed during generated answer evaluation: %s", str(e), exc_info=True)
                # Optionally, populate critiques with error information
                critiques = {"evaluation_error": {"evaluation": str(e), "rating": 1.0}}
                aggregate_score = 1.0 # Assign lowest score on error
        else:
            logger.warning("No evaluator configured. Skipping evaluation of generated answer.")

        # 3. Format and return results
        result = {
            'question': question,
            'generated_answer': generated_answer,
            'ground_truth_answer': ground_truth_answer,
            'critiques': critiques,
            'aggregate_score': aggregate_score
        }

        return result