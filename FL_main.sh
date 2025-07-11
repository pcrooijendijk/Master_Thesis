#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=FL-%j.out
#SBATCH --error=FL-%j.err

# Set environment variables for Hugging Face cache
export HF_HOME="/vol/csedu-nobackup/project/prooijendijk/huggingface/cache"
export HUGGINGFACE_HUB_CACHE="/vol/csedu-nobackup/project/prooijendijk/huggingface/cache"
export TRITON_CACHE_DIR="vol/csedu-nobackup/project/prooijendijk/triton/cache"

# Navigate to the project directory
cd /vol/csedu-nobackup/project/prooijendijk

# Activate the virtual environment (if applicable)
# source myenv/bin/activate

cd /vol/csedu-nobackup/project/prooijendijk/Master_Thesis

# Commands to run your program go here, e.g.:
python FL_main.py 