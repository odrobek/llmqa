"""Critique Agent for evaluating question-answer pairs.

This module provides functionality for evaluating the quality of generated
question-answer pairs using various criteria.
"""

import re
from typing import Dict, Optional
from .base import BaseModel

class CritiqueAgent:
    """Agent for evaluating question-answer pairs using multiple criteria.
    
    This class provides methods to evaluate questions based on:
    - Groundedness: How well can the question be answered from the context
    - Relevance: How useful the question is for ML developers
    - Standalone: How context-independent the question is
    """
    
    # Critique prompts
    GROUNDEDNESS_PROMPT = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}
Context: {context}
Answer::: """

    RELEVANCE_PROMPT = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be for MSOE computer science students in Dr. Lembke's Operating Systems course.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}
Answer::: """

    STANDALONE_PROMPT = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independant this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independant from the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}
Answer::: """
    
    def __init__(self, model: BaseModel):
        """Initialize the critique agent.
        
        Args:
            model (BaseModel): The LLM model to use for critiques
        """
        self.model = model
    
    def _parse_critique(self, response: str) -> Dict[str, str]:
        """Parse the critique response from the model.
        
        Args:
            response (str): Raw response from the model
            
        Returns:
            Dict containing 'rating' and 'evaluation'
            
        Raises:
            ValueError: If response format is invalid
        """
        try:
            # Extract evaluation
            eval_match = re.search(r'Evaluation:\s*(.*?)(?=Total rating:|$)', response, re.DOTALL)
            evaluation = eval_match.group(1).strip() if eval_match else None
            
            # Extract rating
            rating_match = re.search(r'Total rating:\s*(\d+(?:\.\d+)?)', response)
            rating = float(rating_match.group(1)) if rating_match else None
            
            if evaluation is None or rating is None:
                raise ValueError("Could not extract evaluation or rating from response")
                
            return {
                "rating": rating,
                "evaluation": evaluation
            }
            
        except Exception as e:
            raise ValueError(f"Failed to parse critique response: {str(e)}\nResponse: {response}")
    
    def evaluate_groundedness(self, question: str, context: str) -> Dict:
        """Evaluate how well the question can be answered from the context.
        
        Args:
            question (str): The question to evaluate
            context (str): The context from which the question should be answerable
            
        Returns:
            Dict containing rating and evaluation
        """
        prompt = self.GROUNDEDNESS_PROMPT.format(question=question, context=context)
        response = self.model(prompt)
        return self._parse_critique(response)
    
    def evaluate_relevance(self, question: str) -> Dict:
        """Evaluate how useful the question is for ML developers.
        
        Args:
            question (str): The question to evaluate
            
        Returns:
            Dict containing rating and evaluation
        """
        prompt = self.RELEVANCE_PROMPT.format(question=question)
        response = self.model(prompt)
        return self._parse_critique(response)
    
    def evaluate_standalone(self, question: str) -> Dict:
        """Evaluate how context-independent the question is.
        
        Args:
            question (str): The question to evaluate
            
        Returns:
            Dict containing rating and evaluation
        """
        prompt = self.STANDALONE_PROMPT.format(question=question)
        response = self.model(prompt)
        return self._parse_critique(response)
    
    def evaluate_qa_pair(
        self, 
        question: str, 
        context: str,
        criteria: Optional[list] = None
    ) -> Dict:
        """Evaluate a QA pair using specified or all criteria.
        
        Args:
            question (str): The question to evaluate
            context (str): The context from which the question should be answerable
            criteria (list, optional): List of criteria to evaluate. If None, evaluates all.
                Valid criteria: ['groundedness', 'relevance', 'standalone']
                
        Returns:
            Dict containing critiques for each criterion and aggregate score
        """
        all_criteria = ['groundedness', 'relevance', 'standalone']
        criteria = criteria or all_criteria
        
        # Validate criteria
        invalid_criteria = [c for c in criteria if c not in all_criteria]
        if invalid_criteria:
            raise ValueError(f"Invalid criteria: {invalid_criteria}")
        
        # Run evaluations
        critiques = {}
        if 'groundedness' in criteria:
            critiques['groundedness'] = self.evaluate_groundedness(question, context)
        if 'relevance' in criteria:
            critiques['relevance'] = self.evaluate_relevance(question)
        if 'standalone' in criteria:
            critiques['standalone'] = self.evaluate_standalone(question)
            
        # Calculate aggregate score
        ratings = [c['rating'] for c in critiques.values()]
        aggregate_score = sum(ratings) / len(ratings) if ratings else None
        
        return {
            "question": question,
            "critiques": critiques,
            "aggregate_score": aggregate_score
        } 