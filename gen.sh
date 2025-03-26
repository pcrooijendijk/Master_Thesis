#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu-prio,csedu
#SBATCH --qos=csedu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=gen-%j.out
#SBATCH --error=gen-%j.err

# Set environment variables for Hugging Face cache
export HF_HOME=/vol/csedu-nobackup/project/prooijendijk/huggingface/cache

# Navigate to the project directory
cd /vol/csedu-nobackup/project/prooijendijk

# Activate the virtual environment (if applicable)
source myenv/bin/activate

cd /vol/csedu-nobackup/project/prooijendijk/Master_Thesis

# Commands to run your program go here, e.g.:
python generate_response.py
