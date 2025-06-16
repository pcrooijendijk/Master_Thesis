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
    
    # def FedAvg(self, model, selected_clients, dataset_length, epoch, output_dir):
    #     weights_array = torch.tensor([dataset_length[int(cid)] for cid in selected_clients], dtype=torch.float32)
    #     weights_array = torch.nn.functional.normalize(weights_array, p=1, dim=0)

    #     encrypted_weights_dicts = {
    #         cid: self.load_encrypted_weights(
    #             os.path.join(output_dir, str(epoch), f"local_output_{cid}", "encrypted_weights.pkl"),
    #             self.load_public_context("tenseal_context/public_context.tenseal".format(cid))
    #         )
    #         for cid in selected_clients
    #     }

    #     for index, client_id in enumerate(selected_clients):
    #         encrypted_weights = encrypted_weights_dicts[client_id]
    #         weight_scalar = weights_array[index].item()

    #         # Multiply each encrypted chunk by scalar
    #         scaled_encrypted_weights = {
    #             name: [chunk * weight_scalar for chunk in vec]
    #             for name, vec in encrypted_weights.items()
    #         }

    #         if index == 0:
    #             aggregated = scaled_encrypted_weights
    #         else:
    #             aggregated = {
    #                 name: [
    #                     aggregated[name][i] + scaled_encrypted_weights[name][i]
    #                     for i in range(len(scaled_encrypted_weights[name]))
    #                 ]
    #                 for name in aggregated
    #             }

    #     return aggregated

    def get_server_context(self):
        return self.server_context