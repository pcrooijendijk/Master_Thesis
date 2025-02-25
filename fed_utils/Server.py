from torch.nn.functional import normalize
import torch
from peft import (
    set_peft_model_state_dict,
)

class FederatedServer:
    def __init__(self, num_clients, global_model):
        self.global_model = global_model
        self.num_clients = num_clients
    
    def FedAvg(self, model, selected_clients_set, local_dataset):
        weights = normalize(
            torch.tensor([local_dataset[client_id] for client_id in selected_clients_set],
            dtype=torch.float32),
            p=1, dim=0
        )
        for k, client_id in enumerate(selected_clients_set):
            single_output = "model.bin" # Get the models weights
            single_weights = torch.load(single_output)
            if k == 0: 
                weighted_single_weights = {key: single_weights[key] * (weights[k]) for key in single_weights.keys()}
            else: 
                weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights[k])
                                        for key in
                                        single_weights.keys()}
        
        set_peft_model_state_dict(model, weighted_single_weights, "default")

    def run_federated_round(self, clients):
        updates = []
        for client in clients:
            local_update = client.train_local_model()
            if local_update is not None:
                encrypted_update = client.send_encrypted_update()
                updates.append(encrypted_update)

        self.FedAvg(updates)
