# The client should contain the following: name, documents assigned to this client, list of permissions and the local LLM

from perm_utils import Permission
from perm_utils.UserPermissionManagement import UserPermissionsResource

import os
import torch
import torch.nn as nn
import copy
import numpy as np
import transformers
import logging
from typing import List
from collections import OrderedDict
from torch.utils.data import DataLoader
from transformers import TrainerCallback

from sklearn.model_selection import train_test_split
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

np.random.seed(42)
MAX_GRAD_NORM = 0.1

logger = logging.getLogger(__name__)

def client_selection(num_clients, client_frac):
    selected_clients = max(int(client_frac * num_clients), 1)
    return set(np.random.choice(np.arange(num_clients), selected_clients, replace=False))

class DifferentialPrivacyCallback(transformers.TrainerCallback):
    def __init__(self, lora_params, max_grad_norm=1.0, noise_multiplier=1.0):
        self.lora_params = lora_params
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

    def on_step_end(self, args, state, control, **kwargs):
        norms = [p.grad.norm(2) ** 2 for p in self.lora_params if p.grad is not None]

        if not norms:
            return  

        total_norm = torch.sqrt(torch.sum(torch.stack(norms)))
        clip_coef = min(1.0, self.max_grad_norm / (total_norm + 1e-6))

        for p in self.lora_params:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
                noise = torch.normal(
                    mean=0,
                    std=self.noise_multiplier * self.max_grad_norm,
                    size=p.grad.shape,
                    device=p.grad.device,
                )
                p.grad.data.add_(noise)

class Client:
    def __init__(self, client_id: int, name: str, user_permissions_resource: UserPermissionsResource) -> None:
        self.client_id = client_id
        self.name = name
        self.user_permissions_resource = user_permissions_resource

        self.permissions = set()
        self.spaces = set()
        self.rest_user_permission_manager = user_permissions_resource.get_rest_user_permission_manager()
        self.space_manager = self.rest_user_permission_manager.get_space_manager()
        self.documents = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.spaces_permissions_init()
        self.filter_documents()

    def spaces_permissions_init(self) -> None:
        permissions = self.user_permissions_resource.get_permissions(self.name, {"Username": self.name})
        for perm in permissions:
            # Add to the set of spaces this client has access to
            self.spaces.add((perm['spaceName'], perm['spaceKey']))
            for type in perm['permissions']:
                # Add to the set of permissions for this user
                self.permissions.add(type['permissionType']) if type['permissionGranted'] else None

    def filter_documents(self) -> None:
        # Function to filter the documents based on the permissions the clients have
        for space in self.spaces:
            _, space_key = space
            # Only allow to add the documents to the Clients documents if the client has the permission
            # to view these documents
            if Permission.VIEWSPACE_PERMISSION.value in self.permissions:
                self.documents = self.space_manager.get_space(space_key).get_documents()

    def local_dataset_init(self, generate_and_tokenize_prompt) -> None:
        X_train, y_test = train_test_split(
            self.documents, test_size=0.7, shuffle=True
        )
        self.local_train_dataset = list(map(generate_and_tokenize_prompt, X_train))
        self.local_eval_dataset = list(map(generate_and_tokenize_prompt, y_test))
        self.local_train_dataloader = DataLoader(self.local_train_dataset, batch_size=8)
        self.delta = 1 / len(self.local_train_dataset)
    
    def trainer_init(self, tokenizer, accumulation_steps, batch_size, epochs, learning_rate, group_by_length, output_dir) -> None:
        # Use the transformer methods to perform the training steps
        
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size, 
            gradient_accumulation_steps=accumulation_steps,
            warmup_steps=0,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=False,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None,
            group_by_length=group_by_length,
            dataloader_drop_last=False
        )

        # Getting the LoRA parameters
        lora_params = [p for n, p in self.model.named_parameters() if p.requires_grad and "lora_" in n]

        self.local_trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.local_train_dataset,
            eval_dataset=self.local_eval_dataset,
            args=self.train_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            callbacks=[
                DifferentialPrivacyCallback(
                    lora_params=lora_params,
                    max_grad_norm=1.0,
                    noise_multiplier=1.0,
                )
            ]
        )   

    def train(self) -> None:
        # Clear CUDA cache
        torch.cuda.empty_cache()
        self.local_trainer.train()
    
    def local_training(self) -> None:
        self.model.config.use_cache = False
        self.old_params = copy.deepcopy(
            OrderedDict(
                (name, param.detach()) for name, param in self.model.named_parameters() if "default" in name
            )
        )
        self.new_params = OrderedDict(
            (name, param.detach()) for name, param in self.model.named_parameters() if "default" in name
        )
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.new_params, "default"
            )
        ).__get__(self.model, type(self.model))
    
    def end_local_training(self, epoch, dataset_length, selected_clients, output_dir):
        dataset_length[self.client_id] = len(self.documents)
        new_weight = self.model.state_dict()
        output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(self.client_id))
        os.makedirs(output_dir, exist_ok=True)
        torch.save(new_weight, output_dir + "/pytorch_model.bin")

        old_weights = get_peft_model_state_dict(self.model, self.old_params, "default")
        set_peft_model_state_dict(self.model, old_weights, "default")
        last_client_id = self.client_id

        # Clear CUDA cache
        torch.cuda.empty_cache()
        del self.model

        return dataset_length, selected_clients, last_client_id

    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def get_permissions(self):
        return self.permissions

    def get_spaces(self):
        return self.spaces

    def get_documents(self):
        return self.documents
    
    def get_client_id(self) -> int:
        return self.client_id
    
    def set_model(self, model) -> None: 
        self.model = model

def __getstate__(self):
    return {
        "client_id": self.client_id,
        "name": self.name,
        "permissions": self.permissions,
        "spaces": self.spaces,
        "documents": self.documents,
    }

def __setstate__(self, state):
    self.client_id = state["client_id"]
    self.name = state["name"]
    self.permissions = state["permissions"]
    self.spaces = state["spaces"]
    self.documents = state["documents"]
    
    self.user_permissions_resource = None
    self.rest_user_permission_manager = None
    self.space_manager = None
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")