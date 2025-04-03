import torch
import os
from torch.nn.functional import normalize
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

class Server:
    def __init__(self, num_clients, global_model):
        self.global_model = global_model
        self.num_clients = num_clients
    
    def model_init(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.local_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model)
        self.tokenizer.pad_token_id = (
            0
        )
        self.tokenizer.padding_side = "left"

        # Using this technique to reduce memory-usage and accelarting inference
        self.model = prepare_model_for_kbit_training(self.model) 

        # Initialize LoRA
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=16, 
            target_modules=[ # The layers which need to be finetuned
                "q_proj",
            ], 
            lora_dropout=0.05, 
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Get the PEFT model using LoRA
        self.model = get_peft_model(self.model, lora_config)
    
    def FedAvg(self, selected_clients, dataset_length, epoch, output_dir):
        # Normalizing the weights of each client
        # TODO: change the following to the pytorch implementation
        self.model_init()
        weights_array = normalize(
            torch.tensor([dataset_length[int(client_id)] for client_id in selected_clients],
                        dtype=torch.float32), p=1, dim=0)

        for k, client_id in enumerate(selected_clients):
            total_output_dir = os.path.join(
                output_dir, str(epoch), "local_output_{}".format(client_id), "pytorch_model.bin"
            )
            # Get the weights from the client from the output directory
            weights = torch.load(total_output_dir)
            if k == 0:
                weighted_weights = {key: weights[key] * (weights_array[k]) for key in
                                        weights.keys()}
            else:
                weighted_weights = {key: weighted_weights[key] + weights[key] * (weights_array[k])
                                        for key in weights.keys()}

        set_peft_model_state_dict(self.model, weighted_weights, "default")

        return self.model