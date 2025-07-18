import torch
import os
import pickle
import tenseal as ts

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
            poly_modulus_degree=32768,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        return context
    
    def load_encrypted_weights(self, input_path, context):
        with open(input_path, 'rb') as f:
            serialized = pickle.load(f)
            encrypted_weights = {
                k: [ts.ckks_vector_from(context, chunk) for chunk in v]
                for k, v in serialized.items()
            }
        return encrypted_weights
    
    def FedAvg(self, model, selected_clients, dataset_length, epoch, output_dir):
        weights_array = torch.tensor([dataset_length[int(cid)] for cid in selected_clients], dtype=torch.float32)
        weights_array = torch.nn.functional.normalize(weights_array, p=1, dim=0)

        encrypted_weights_dicts = {
            cid: self.load_encrypted_weights(
                os.path.join(output_dir, str(epoch), f"local_output_{cid}", "encrypted_weights.pkl"),
                self.server_context
            )
            for cid in selected_clients
        }

        for index, client_id in enumerate(selected_clients):
            encrypted_weights = encrypted_weights_dicts[client_id]
            weight_scalar = weights_array[index].item()

            # Multiply each encrypted chunk by scalar
            scaled_encrypted_weights = {
                name: [chunk * weight_scalar for chunk in vec]
                for name, vec in encrypted_weights.items()
            }

            if index == 0:
                aggregated = scaled_encrypted_weights
            else:
                aggregated = {
                    name: [
                        aggregated[name][i] + scaled_encrypted_weights[name][i]
                        for i in range(len(scaled_encrypted_weights[name]))
                    ]
                    for name in aggregated
                }

        return aggregated

    
    def get_server_context(self):
        return self.server_context