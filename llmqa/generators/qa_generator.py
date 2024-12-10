"""QA Generator implementation.

This module provides functionality for generating question-answer pairs from text chunks
using LLM models.
"""

import pandas as pd
import json
import ast
import copy
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Type
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import functools

from ..models.base import BaseModel
from ..models.critique_agent import CritiqueAgent

def _process_chunk_worker(args: Tuple[Type[BaseModel], Type[CritiqueAgent], str, float, List[str], int]) -> List[Dict[str, Any]]:
    """Worker function to process a chunk in a separate process.
    
    Args:
        args: Tuple containing:
            - model_class: Class of the QA model
            - critique_class: Class of the critique model (or None)
            - chunk: Text chunk to process
            - min_score: Minimum score threshold
            - criteria: List of critique criteria
            - max_retries: Maximum number of retries
            
    Returns:
        List[Dict[str, Any]]: Generated QA pairs with optional critique information
    """
    model_class, critique_class, chunk, min_score, criteria, max_retries = args
    
    # Create fresh model instances in this process
    model = model_class()  # QA model
    critique_model = model_class()  # Separate model instance for critique
    critique_agent = critique_class(critique_model) if critique_class else None
    
    for attempt in range(max_retries):
        try:
            response = model(chunk)
            qa_pairs = ast.literal_eval(response)
            
            # Validate response format
            if not isinstance(qa_pairs, list):
                raise ValueError("Model response must be a list")
            
            processed_pairs = []
            for pair in qa_pairs:
                if not isinstance(pair, dict) or 'question' not in pair or 'answer' not in pair:
                    raise ValueError("Each QA pair must be a dict with 'question' and 'answer' keys")
                
                # Add source context to the pair
                processed_pair = {
                    **pair,
                    "source_context": chunk
                }
                
                # If we have a critique agent, evaluate the pair
                if critique_agent:
                    critique_result = critique_agent.evaluate_qa_pair(
                        question=pair['question'],
                        context=chunk,
                        criteria=criteria
                    )
                    processed_pair['critiques'] = critique_result['critiques']
                    processed_pair['aggregate_score'] = critique_result['aggregate_score']
                    
                    # Only include pairs that meet the minimum score
                    if processed_pair['aggregate_score'] >= min_score:
                        processed_pairs.append(processed_pair)
                else:
                    processed_pairs.append(processed_pair)
            
            return processed_pairs
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise ValueError(
                    f"Failed to generate valid QA pairs after {max_retries} attempts. "
                    f"Last error: {str(e)}"
                )
            # Add context about the error to help the model
            chunk = f"You just tried to create a response for the following chunk but got this error {e}. Make sure your new response does not cause this.\n{chunk}"

class QAGenerator:
    """Generator for question-answer pairs from text chunks.
    
    This class handles the generation of QA pairs using a specified LLM model.
    
    Example:
        >>> from llmqa import ROSIELlama, QAGenerator
        >>> model = ROSIELlama()
        >>> generator = QAGenerator(model)
        >>> qa_pairs = generator.generate_from_chunk("Your text here")
        >>> print(qa_pairs)
    """
    
    def __init__(
        self, 
        model: BaseModel, 
        num_workers: Optional[int] = None,
        critique_agent: Optional[CritiqueAgent] = None,
        min_critique_score: float = 3.0,
        critique_criteria: Optional[List[str]] = None
    ):
        """Initialize the QA Generator.
        
        Args:
            model (BaseModel): The LLM model to use for generation
            num_workers (int, optional): Number of worker processes. If None,
                uses CPU count - 1. Set to 1 to disable multiprocessing.
            critique_agent (CritiqueAgent, optional): Agent for evaluating QA pairs
            min_critique_score (float): Minimum aggregate score to keep a QA pair
            critique_criteria (List[str], optional): Criteria to evaluate. If None,
                uses all criteria ['groundedness', 'relevance', 'standalone']
        """
        self.model = model
        self.model_class = model.__class__  # Store the class for creating new instances
        self.num_workers = num_workers if num_workers is not None else max(1, multiprocessing.cpu_count() - 1)
        self.critique_agent = critique_agent
        self.critique_class = critique_agent.__class__ if critique_agent else None
        self.min_critique_score = min_critique_score
        self.critique_criteria = critique_criteria
    
    def generate_from_chunk(
        self, 
        chunk: str, 
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate QA pairs from a single text chunk.
        
        Args:
            chunk (str): The text chunk to generate QA pairs from
            max_retries (int): Maximum number of retries on parsing errors
            
        Returns:
            List[Dict[str, Any]]: List of QA pairs with optional critique information
            
        Raises:
            ValueError: If chunk is empty or if unable to generate valid QA pairs
            TypeError: If chunk is not a string
        """
        if not isinstance(chunk, str):
            raise TypeError(f"Chunk must be a string, got {type(chunk)}")
        if not chunk.strip():
            raise ValueError("Chunk cannot be empty")
            
        return _process_chunk_worker((
            self.model_class,
            self.critique_class,
            chunk,
            self.min_critique_score,
            self.critique_criteria,
            max_retries
        ))
    
    def generate_from_file(
        self, 
        input_file: Union[str, Path], 
        output_file: Union[str, Path],
        chunk_column: str = 'processed_text',
        verbose: bool = False,
        append: bool = True,
        batch_size: int = 10
    ) -> None:
        """Generate QA pairs from a CSV file containing text chunks.
        
        Args:
            input_file (str | Path): Path to input CSV file
            output_file (str | Path): Path to output JSON file
            chunk_column (str): Name of the column containing text chunks
            verbose (bool): Whether to print progress messages
            append (bool): Whether to append to existing output file or overwrite
            batch_size (int): Number of chunks to process in parallel
        """
        # Convert to Path objects for better path handling
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        # Validate input file
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read chunks from CSV
        try:
            chunks = pd.read_csv(input_path)
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(f"Input file is empty: {input_file}")
            
        if chunk_column not in chunks.columns:
            raise ValueError(
                f"Column '{chunk_column}' not found in input file. "
                f"Available columns: {', '.join(chunks.columns)}"
            )
        
        # Initialize or load existing data
        existing_data = []
        if append and output_path.exists():
            try:
                with open(output_path) as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                if verbose:
                    print(f"Warning: Could not read existing output file {output_file}, starting fresh")
        
        all_qa_pairs = existing_data
        total_generated = 0
        total_filtered = 0
        
        # Process chunks in parallel
        if verbose:
            print(f"Processing {len(chunks)} chunks using {self.num_workers} workers", flush=True)
            if self.critique_agent:
                print(f"Using critique agent with minimum score: {self.min_critique_score}", flush=True)
                if self.critique_criteria:
                    print(f"Evaluating criteria: {', '.join(self.critique_criteria)}", flush=True)
        
        # Process in batches to avoid memory issues
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch = chunks.iloc[batch_start:batch_end]
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all chunks in this batch
                future_to_chunk = {
                    executor.submit(
                        _process_chunk_worker,
                        (
                            self.model_class,
                            self.critique_class,
                            row[chunk_column],
                            self.min_critique_score,
                            self.critique_criteria,
                            3
                        )
                    ): i 
                    for i, row in batch.iterrows()
                }
                
                # Process completed chunks with progress bar
                with tqdm(total=len(future_to_chunk), disable=not verbose) as pbar:
                    for future in as_completed(future_to_chunk):
                        try:
                            qa_pairs = future.result()
                            total_generated += len(qa_pairs)
                            all_qa_pairs.extend(copy.deepcopy(qa_pairs))
                            
                            # Save after each chunk
                            with open(output_path, 'w') as f:
                                json.dump(all_qa_pairs, f, indent=4)
                                
                            if verbose:
                                desc = f"Generated {len(qa_pairs)} pairs"
                                if self.critique_agent:
                                    filtered = sum(1 for p in qa_pairs if p['aggregate_score'] >= self.min_critique_score)
                                    total_filtered += filtered
                                    desc += f" (kept {filtered})"
                                pbar.set_description(desc)
                            
                        except Exception as e:
                            chunk_idx = future_to_chunk[future]
                            print(f"Error processing chunk {chunk_idx}: {str(e)}")
                        
                        pbar.update(1)
            
            if verbose:
                print(f"Completed batch {batch_start//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}", flush=True)
                
        if verbose and self.critique_agent:
            print(f"\nGeneration complete!")
            print(f"Total QA pairs generated: {total_generated}")
            print(f"QA pairs meeting critique threshold: {total_filtered}")
            print(f"Filtering rate: {(total_generated - total_filtered) / total_generated:.1%}")