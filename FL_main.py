from fed_utils import Client, client_selection, Server
from utils import Dataset, Document, SpaceManagement, PromptHelper, Users

import torch
import fire
from typing import List
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

documents = load_dataset("json", data_files="utils/documents.json")

# Intialize the spaces
space_names = ["mark", "new", "dev", "HR"]
space_keys = [0, 1, 2, 3]

users = Users(space_keys).get_users()

# Initialize the spaces, space manager, user accessor, user manager and the space permission manager by using a space management
management = SpaceManagement(space_names, space_keys, documents, users)
user_permissions_resource = management.get_user_permissions_resource()

# Main federated learning function
def federated_privacy_learning(
    global_model: str = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', # The global model
    output_dir: str = 'FL_output/', # The output directory
    client_frac: float = 0.1, # The fraction of clients chosen from the total number of clients
    comm_rounds: int = 10, # Number of communication rounds
    num_clients: int = 2, # Number of clients
    batch_size = 4, # Batch size for the local models
    micro_batch_size: int = 1, # Micro batch size for the local models
    epochs: int = 1, # Number of total epochs for the local models to train on
    lr: float = 1e-2, # Learning rate for the local models
    save_steps: int = 3, # After this amount of steps there is a checkpoint
    max_length: int = 512, # After this length there is a cutoff 
    lora_rank: int = 16 , # Lora attention dimension 
    lora_alpha: int = 16, # The alpha parameter for Lora scaling
    lora_dropout: float = 0.05, # The dropout probability for Lora layers
    lora_module: List[str] = [ # The layers which need to be finetuned
        "q_proj",
    ],
    training_on_inputs: bool = True, 
    group_by_length: bool = False,
    template: str = 'utils/prompt_template.json', # Prompt template 
):
    assert global_model, "Please specify a global model, for instance: deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    gradient_steps = batch_size // micro_batch_size
    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        global_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(global_model)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"

    # Helper functions for the training process
    def tokenizer_init(prompt: str, add_eos_token: bool=True):
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            if (
                    result["input_ids"][-1] != tokenizer.eos_token_id
                    and len(result["input_ids"]) < max_length
                    and add_eos_token
            ):
                result["input_ids"].append(tokenizer.eos_token_id)
                result["attention_mask"].append(1)

            result["labels"] = result["input_ids"].copy()

            return result

    def generate_and_tokenize_prompt(document: dict):
            prompt_helper = PromptHelper(template)
            full_prompt = prompt_helper.generate_prompt(
                document["question"],
                document["context"]
            )
            tokenized_full_prompt = tokenizer_init(full_prompt)
            return tokenized_full_prompt

    # Using this technique to reduce memory-usage and accelarting inference
    model = prepare_model_for_kbit_training(model) 

    # Initialize LoRA
    lora_config = LoraConfig(
        r=lora_rank, 
        lora_alpha=lora_alpha, 
        target_modules=lora_module, 
        lora_dropout=lora_dropout, 
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get the PEFT model using LoRA
    model = get_peft_model(model, lora_config)

    # Initialize before the federated learning starts
    selected_clients = set()
    last_client = None
    dataset_length = dict()

    for epoch in tqdm(range(comm_rounds)):
        print("Selecting clients...")
        # Selecting the indices of the clients which will be used for FL 
        selected_clients_index = client_selection(num_clients, client_frac)

        # Get all the clients
        clients = Users.get_clients()

        # Get the correct client IDs from all the clients
        selected_clients = [clients[index].get_client_id() for index in selected_clients_index]

        # Initialize the server
        server = Server(num_clients=len(clients), global_model=global_model)

        for client_id in selected_clients_index: 
            client = clients[client_id] # TODO: Fix this according to the clients list!
            print("\nPreparing the local dataset and trainter for client {}".format(client_id))
            client.local_dataset_init(generate_and_tokenize_prompt)
            client.trainer_init(
                tokenizer, 
                micro_batch_size, 
                batch_size, 
                epochs, 
                lr, 
                group_by_length,
                output_dir
            )

            print("\nInitializing the local training of client {}".format(client_id))
            client.local_training()

            print("\nStarting local training...")
            client.train()

            print("\nEnding the local training of client {}".format(client_id))
            model, dataset_length, selected_clients, last_client = client.end_local_training(
                epoch, dataset_length, selected_clients, output_dir
                )
        
        print('\nGetting the weights of the clients and send it to the server for aggregation')
        model = server.FedAvg(model, selected_clients, dataset_length, epoch, output_dir)
        torch.save(model.state_dict(), output_dir + "pytorch_model.bin")
        lora_config.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(federated_privacy_learning)