import torch
import os
import gc
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

    def load_encrypted_update(self, path, context):
        with open(path, "rb") as f:
            return ts.ckks_vector_from(context, f.read())
    
    def FedAvg(self, model, selected_clients, dataset_length, epoch, output_dir):
        encrypted_paths = [f"'FL_output/' + str{self.client_id} + encrypted_weights.pkl" for id in selected_clients]
        context = ts.context_from(open("tenseal_public_context.tenseal", "rb").read())

        # Normalizing the weights of each client
        weights_array = normalize(
            torch.tensor([dataset_length[int(client_id)] for client_id in selected_clients],
                        dtype=torch.float32), p=1, dim=0)
        
        weighted_sum = None
        for index, client_id in enumerate(selected_clients):
            path = encrypted_paths[index]

            with open(path, "rb") as f:
                encrypted_weights = ts.ckks_tensor_from(context, f.read())

            encrypted_weights *= weights_array[index].item()

            if weighted_sum is None:
                weighted_sum = encrypted_weights
            else:
                weighted_sum += encrypted_weights
        return weighted_sum

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
    
    def get_server_context(self):
        return self.server_context