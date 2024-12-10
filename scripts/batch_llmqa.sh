#!/bin/bash

################################################################################
#
# LLMQA Generation Batch Job
#
# This script runs question-answer generation on a CSV file of text chunks using
# the LLMQA library and ROSIE's LLaMA model.
#
# Before running:
# 1. Make sure you have the conda environment set up:
#    conda env create -f environment.yml
#
# 2. Set up your ROSIE credentials in the conda environment:
#    cd <conda_env_path>
#    mkdir -p ./etc/conda/activate.d
#    mkdir -p ./etc/conda/deactivate.d
#    
#    # In ./etc/conda/activate.d/env_vars.sh:
#    export ROSIE_URL='<URL>'
#    export ROSIE_KEY='<KEY>'
#    
#    # In ./etc/conda/deactivate.d/env_vars.sh:
#    unset ROSIE_URL
#    unset ROSIE_KEY
#
# 3. Edit the configuration variables below
#
# To submit: sbatch batch_llmqa.sh
# To monitor: squeue -l -u $USER
# To cancel: scancel <jobid>
#
################################################################################

#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gpus=4  # Request multiple GPUs if available
#SBATCH --cpus-per-gpu=8  # Adjusted CPU per GPU ratio
#SBATCH --error='sbatcherrorfile.out'
#SBATCH --time=0-1:0
#SBATCH --mem=32G  # Request more memory for parallel processing

############################
# Configuration
############################

# Input/Output Settings
INPUT_FILE="<INPUT_FILE>.csv"      # Path to your input CSV file
OUTPUT_FILE="<OUTPUT_FILE>.json"    # Where to save the QA pairs
CHUNK_COLUMN="processed_text"       # Column name containing text chunks
VERBOSE="True"                      # Whether to show detailed progress

# Performance Settings
NUM_WORKERS=8                       # Number of parallel workers (should match --cpus-per-gpu)
BATCH_SIZE=20                       # Number of chunks to process in parallel

# Validate configuration
if [[ "$INPUT_FILE" == "<INPUT_FILE>.csv" ]]; then
    echo "Error: Please edit INPUT_FILE in the script before running"
    exit 1
fi

if [[ "$OUTPUT_FILE" == "<OUTPUT_FILE>.json" ]]; then
    echo "Error: Please edit OUTPUT_FILE in the script before running"
    exit 1
fi

# Get script locations
SCRIPT_NAME="llmqa_generation"
PYTHON_SCRIPT="generate_qa.py"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

############################
# Environment Setup
############################

echo "Setting up environment..."

# Activate conda
source activate llmqa

# Check for ROSIE credentials
if [[ -z "${ROSIE_URL}" ]] || [[ -z "${ROSIE_KEY}" ]]; then
    echo "Error: ROSIE_URL and ROSIE_KEY environment variables must be set"
    exit 1
fi

# Install package if needed
if ! python -c "import llmqa" &> /dev/null; then
    echo "Installing llmqa package..."
    cd "$PROJECT_ROOT/LLMQA"
    pip install -e .
fi

############################
# Generate Python Script
############################

echo "Creating generation script..."

cat > ${PYTHON_SCRIPT} << 'EOL'
from llmqa import ROSIELlama, QAGenerator
import argparse
from pathlib import Path
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Generate QA pairs from text chunks")
    parser.add_argument('--input_file', type=str, required=True,
                      help="Path to input CSV file with text chunks")
    parser.add_argument('--output_file', type=str, required=True,
                      help="Path to output JSON file for QA pairs")
    parser.add_argument('--chunk_column', type=str, default='processed_text',
                      help="Name of the column containing text chunks")
    parser.add_argument('--verbose', type=bool, default=True,
                      help="Whether to print progress messages")
    parser.add_argument('--num_workers', type=int, default=None,
                      help="Number of worker processes")
    parser.add_argument('--batch_size', type=int, default=10,
                      help="Number of chunks to process in parallel")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    # Initialize model and generator
    print("Initializing ROSIE LLaMA model...")
    model = ROSIELlama()
    generator = QAGenerator(model, num_workers=args.num_workers)
    
    # Generate QA pairs
    print(f"\nProcessing chunks from {args.input_file}")
    print(f"Output will be saved to {args.output_file}")
    print(f"Using {generator.num_workers} worker processes")
    print(f"Batch size: {args.batch_size}\n")
    
    generator.generate_from_file(
        args.input_file,
        args.output_file,
        args.chunk_column,
        args.verbose,
        batch_size=args.batch_size
    )
    
    print("\nGeneration complete!")

if __name__ == "__main__":
    main()
EOL

############################
# Run Generation
############################

echo "Starting ${SCRIPT_NAME} job..."
echo "Input file: ${INPUT_FILE}"
echo "Output file: ${OUTPUT_FILE}"
echo "Chunk column: ${CHUNK_COLUMN}"
echo "Number of workers: ${NUM_WORKERS}"
echo "Batch size: ${BATCH_SIZE}"
echo

srun hostname; pwd; date

srun python3 -u ${PYTHON_SCRIPT} \
    --input_file "${INPUT_FILE}" \
    --output_file "${OUTPUT_FILE}" \
    --chunk_column "${CHUNK_COLUMN}" \
    --verbose "${VERBOSE}" \
    --num_workers "${NUM_WORKERS}" \
    --batch_size "${BATCH_SIZE}"

echo
echo "Job finished!"
echo "Check ${OUTPUT_FILE} for results"
echo "Check sbatcherrorfile.out for any errors"

# Clean up
rm ${PYTHON_SCRIPT}