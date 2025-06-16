import torch
import os
import pickle
import tenseal as ts

class Server:
    def __init__(self, num_clients, global_model):
        self.global_model = global_model
        self.num_clients = num_clients
    
    def load_encrypted_weights(self, input_path, context):
        with open(input_path, 'rb') as f:
            serialized = pickle.load(f)
            encrypted_weights = {
                k: [ts.ckks_vector_from(context, chunk) for chunk in v]
                for k, v in serialized.items()
            }
        return encrypted_weights
    
    def load_public_context(self, context_path):
        with open(context_path, "rb") as f:
            return ts.context_from(f.read()) 
        
    def FedAvg(self, encrypted_updates, context):
        aggregated_update = {}
        for k in encrypted_updates[0].keys():
            encrypted_sum = encrypted_updates[0][k]
            for update in encrypted_updates[1:]:
                encrypted_sum += update[k]
            encrypted_avg = encrypted_sum * (1.0 / len(encrypted_updates))
            aggregated_update[k] = encrypted_avg
        return aggregated_update

    def get_server_context(self):
        return self.server_context