from torch.nn.functional import normalize
import torch
import os
from peft import (
    set_peft_model_state_dict,
)

class FederatedServer:
    def __init__(self, num_clients, global_model):
        self.global_model = global_model
        self.num_clients = num_clients
    
    def FedAvg(self, model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):
        weights_array = normalize(
            torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                        dtype=torch.float32),
            p=1, dim=0)

        for k, client_id in enumerate(selected_clients_set):
            single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
                                            "pytorch_model.bin")
            single_weights = torch.load(single_output_dir)
            if k == 0:
                weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in
                                        single_weights.keys()}
            else:
                weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                        for key in
                                        single_weights.keys()}

        set_peft_model_state_dict(model, weighted_single_weights, "default")

        return model
    
    # def FedAvg(self, model, selected_clients_set, local_dataset):
    #     self.weights = normalize(
    #         torch.tensor([local_dataset[client_id] for client_id in selected_clients_set],
    #         dtype=torch.float32),
    #         p=1, dim=0
    #     )
    #     for k, client_id in enumerate(selected_clients_set):
    #         single_output = "model.bin" # Get the models weights
    #         single_weights = torch.load(single_output)
    #         if k == 0: 
    #             weighted_single_weights = {key: single_weights[key] * (self.weights[k]) for key in single_weights.keys()}
    #         else: 
    #             weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (self.weights[k])
    #                                     for key in
    #                                     single_weights.keys()}
        
    #     set_peft_model_state_dict(model, weighted_single_weights, "default")

    # def run_federated_round(self, clients):
    #     updates = []
    #     for client in clients:
    #         local_update = client.train_local_model()
    #         if local_update is not None:
    #             encrypted_update = client.send_encrypted_update()
    #             updates.append(encrypted_update)

    #     self.FedAvg(updates)
    
    def get_weights(self):
        pass

# class InferDPTServer(object):

#     def __init__(self, ctx: Context, inference_inst: Inference) -> None:
        
#         self.ctx = ctx
#         self.inference_inst = inference_inst
#         self.comm_idx = 0 

#     def inference(self, verbose=False):

#         client_data = self.ctx.guest.get('client_data_{}'.format(self.comm_idx))
#         perturbed_docs, inference_kwargs = client_data

#         if verbose:
#             logger.info('got data {}'.format(client_data))

#         logger.info('start inference')
#         rs_doc = self.inference_inst.inference(perturbed_docs, inference_kwargs)
#         self.ctx.guest.put('pdoc_{}'.format(self.comm_idx), rs_doc)
#         self.comm_idx += 1

#     def predict(self):
#         self.inference()
