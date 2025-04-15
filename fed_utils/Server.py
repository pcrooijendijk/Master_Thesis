import torch
import os
import gc
from torch.nn.functional import normalize
from peft import (
    set_peft_model_state_dict,
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

        for k, client_id in enumerate(selected_clients):
            total_output_dir = os.path.join(
                output_dir, str(epoch), f"local_output_{client_id}", "pytorch_model.bin"
            )

            # Load weights on CPU, not GPU
            weights = torch.load(total_output_dir, map_location='cpu')

            if k == 0:
                weighted_weights = {key: weights[key] * weights_array[k] for key in weights}
            else:
                weighted_weights = {
                    key: weighted_weights[key] + weights[key] * weights_array[k]
                    for key in weights
                }

            # Clean up to avoid GPU memory issues
            del weights
            gc.collect()
            torch.cuda.empty_cache()


        set_peft_model_state_dict(model, weighted_weights, "default")

        return model