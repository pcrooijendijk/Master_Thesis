import torch
import os
import gc
import pickle
import tenseal as ts
from torch.nn.functional import normalize
from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import (
    set_peft_model_state_dict,
)

class Server:
    def __init__(self, num_clients, global_model):
        self.global_model = global_model
        self.num_clients = num_clients

        self.server_context = self.generate_context()

        with open("tenseal_full_context.tenseal", "wb") as f:
            f.write(self.server_context.serialize(save_secret_key=True))

        with open("tenseal_public_context.tenseal", "wb") as f:
            f.write(self.server_context.serialize(save_secret_key=False))

    def generate_context(self):
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        return context
    
    def load_public_context(self):
        with open("tenseal_public_context.tenseal", "rb") as f:
            return ts.context_from(f.read())

    def load_encrypted_weights(self, file_path, context):
        with open(file_path, 'rb') as f:
            serialized = pickle.load(f)
            return {k: ts.ckks_vector_from(context, v) for k, v in serialized.items()}
    
    def FedAvg(self, model, selected_clients, dataset_length, epoch, output_dir):
        weights_array = torch.tensor([dataset_length[int(cid)] for cid in selected_clients], dtype=torch.float32)
        weights_array = torch.nn.functional.normalize(weights_array, p=1, dim=0)

        encrypted_weights_dicts = {
            cid: self.load_encrypted_weights(f"FL_output/" + str(cid) + "encrypted_weights.pkl", self.server_context)
            for cid in selected_clients
        }

        for index, client_id in enumerate(selected_clients):
            encrypted_weights = encrypted_weights_dicts[client_id]
            weight_scalar = weights_array[index].item()

            # Scale each encrypted vector in the dict
            scaled_encrypted_weights = {
                name: vec * weight_scalar
                for name, vec in encrypted_weights.items()
            }

            if index == 0:
                aggregated = scaled_encrypted_weights
            else:
                aggregated = {
                    name: aggregated[name] + scaled_encrypted_weights[name]
                    for name in aggregated
                }

        return aggregated  # Still encrypted
        # encrypted_paths = [f"FL_output/" + str(id) + "encrypted_weights.pkl" for id in selected_clients]
        # context = ts.context_from(open("tenseal_public_context.tenseal", "rb").read())

        # # Normalizing the weights of each client
        # weights_array = normalize(
        #     torch.tensor([dataset_length[int(client_id)] for client_id in selected_clients],
        #                 dtype=torch.float32), p=1, dim=0)
        
        # weighted_sum = None
        # for index, client_id in enumerate(selected_clients):
        #     path = encrypted_paths[index]

        #     with open(path, "rb") as f:
        #         serialized = pickle.load(f)
        #         encrypted_weights =  {k: ts.ckks_vector_from(context, v) for k, v in serialized.items()}

        #     scaled_encrypted_weights = {
        #         name: vec * weights_array[index].item()
        #         for name, vec in encrypted_weights.items()
        #     }

        #     if index == 0:
        #         aggregated_encrypted_weights = scaled_encrypted_weights
        #     else:
        #         aggregated_encrypted_weights = {
        #             name: aggregated_encrypted_weights[name] + scaled_encrypted_weights[name]
        #             for name in aggregated_encrypted_weights
        #         }

        # return weighted_sum
    
    def get_server_context(self):
        return self.server_context