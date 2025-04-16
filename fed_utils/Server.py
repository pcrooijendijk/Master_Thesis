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
    
    def FedAvg(self, model, selected_clients, dataset_length, epoch, output_dir):
        # Normalizing the weights of each client
        # TODO: change the following to the pytorch implementation
        weights_array = normalize(
            torch.tensor([dataset_length[int(client_id)] for client_id in selected_clients],
                        dtype=torch.float32), p=1, dim=0)

        with torch.no_grad():
            for index, client_id in enumerate(selected_clients):
                total_output_dir = os.path.join(
                    output_dir, str(epoch), f"local_output_{client_id}", "pytorch_model.bin"
                )

                weights = torch.load(total_output_dir)
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

        set_peft_model_state_dict(model, weighted_weights, "default")

        return model