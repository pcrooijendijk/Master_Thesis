import torch
import os
from torch.nn.functional import normalize
from peft import (
    set_peft_model_state_dict,
)

class Server:
    def __init__(self, num_clients, global_model):
        self.global_model = global_model
        self.num_clients = num_clients
    
    def FedAvg(self, selected_clients, dataset_length, epoch, output_dir):
        # Normalizing the weights of each client
        # TODO: change the following to the pytorch implementation
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
            torch.cuda.empty_cache()

        set_peft_model_state_dict(self.model, weighted_weights, "default")

        return self.model