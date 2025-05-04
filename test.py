from fed_utils import client_selection, Server
from utils import SpaceManagement, PromptHelper, Users

import torch
import fire
import pickle
from typing import List
import argparse
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datasets import load_dataset

global_model = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
local_model = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
output_dir = 'FL_output/'

all_documents = load_dataset("json", data_files="utils/documents.json")
temp_doc = []
for index, _ in enumerate(all_documents['train']):
    temp_doc.append(all_documents["train"][index])

print(temp_doc)