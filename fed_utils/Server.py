import torch
import os
import gc
from torch.nn.functional import normalize
from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
)

class Server:
    def __init__(self, num_clients, global_model):
        self.global_model = global_model
        self.num_clients = num_clients
    
    def setting_peft_model(self) -> None:
        model = AutoModelForCausalLM.from_pretrained(
            self.global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map="cpu",
            )

        tokenizer = AutoTokenizer.from_pretrained(self.global_model)
        tokenizer.pad_token_id = (
            0
        )
        tokenizer.padding_side = "left"

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
        self.model = get_peft_model(model, lora_config)
    
    def FedAvg(self, model, selected_clients, dataset_length, epoch, output_dir):
        # Normalizing the weights of each client
        # TODO: change the following to the pytorch implementation
        weights_array = normalize(
            torch.tensor([dataset_length[int(client_id)] for client_id in selected_clients],
                        dtype=torch.float32), p=1, dim=0)

        for index, client_id in enumerate(selected_clients):
            total_output_dir = os.path.join(
                output_dir, str(epoch), f"local_output_{client_id}", "pytorch_model.bin"
            )

            # Load weights on CPU, not GPU
            # weights = torch.load(total_output_dir, map_location=torch.device("cpu"))
            weights = torch.load(total_output_dir)

            # for k in weights: 
            #     weights[k] = weights[k].float()

            if index == 0:
                weighted_weights = {key: weights[key] * weights_array[index] for key in weights}
            else:
                weighted_weights = {
                    key: weighted_weights[key] + weights[key] * weights_array[index]
                    for key in weights
                }

            # Clean up to avoid GPU memory issues
            del weights
            gc.collect()
            torch.cuda.empty_cache()

        # for key in weighted_weights:
        #     weighted_weights[key] = weighted_weights[key].float().cpu()

        self.setting_peft_model()    

        set_peft_model_state_dict(self.model, weighted_weights, "default")

        return model