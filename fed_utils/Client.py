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
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.model_selection import train_test_split
from opacus import PrivacyEngine
from opacus.grad_sample.grad_sample_module import GradSampleModule
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig,
)

np.random.seed(42)
MAX_GRAD_NORM = 0.1

logger = logging.getLogger(__name__)

def client_selection(num_clients, client_frac):
    selected_clients = max(int(client_frac * num_clients), 1)
    return set(np.random.choice(np.arange(num_clients), selected_clients, replace=False))

class Client:
    def __init__(self, client_id: int, name: str, user_permissions_resource: UserPermissionsResource, model) -> None:
        self.client_id = client_id
        self.name = name
        self.user_permissions_resource = user_permissions_resource
        self.model = model

        self.permissions = set()
        self.spaces = set()
        self.rest_user_permission_manager = user_permissions_resource.get_rest_user_permission_manager()
        self.space_manager = self.rest_user_permission_manager.get_space_manager()
        self.documents = []
        self.privacy_engine = PrivacyEngine()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.spaces_permissions_init()
        self.filter_documents()
    
    # def model_init(
    #     self, 
    #     lora_rank: int = 16 , # Lora attention dimension 
    #     lora_alpha: int = 16, # The alpha parameter for Lora scaling
    #     lora_dropout: float = 0.05, # The dropout probability for Lora layers
    #     lora_module: List[str] = [ # The layers which need to be finetuned
    #         "q_proj",
    #     ],
    # )-> None:
    #     self.model = AutoModelForCausalLM.from_pretrained(
    #         self.local_model,
    #         load_in_8bit=True,
    #         torch_dtype=torch.float16,
    #         device_map="auto",
    #         llm_int8_enable_fp32_cpu_offload=True,
    #     )

    #     self.tokenizer = AutoTokenizer.from_pretrained(self.local_model)
    #     self.tokenizer.pad_token_id = (
    #         0
    #     )
    #     self.tokenizer.padding_side = "left"

    #     # Using this technique to reduce memory-usage and accelarting inference
    #     self.model = prepare_model_for_kbit_training(self.model) 

    #     # Initialize LoRA
    #     lora_config = LoraConfig(
    #         r=lora_rank, 
    #         lora_alpha=lora_alpha, 
    #         target_modules=lora_module, 
    #         lora_dropout=lora_dropout, 
    #         bias="none",
    #         task_type="CAUSAL_LM"
    #     )

        # # Get the PEFT model using LoRA
        # self.model = get_peft_model(self.model, lora_config)
    
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
        # self.local_train_dataset = list(map(lambda x: generate_and_tokenize_prompt(x, self.tokenizer), X_train))
        # self.local_eval_dataset = list(map(lambda x: generate_and_tokenize_prompt(x, self.tokenizer), y_test))
        self.local_train_dataset = list(map(generate_and_tokenize_prompt, X_train))
        self.local_eval_dataset = list(map(generate_and_tokenize_prompt, y_test))
        self.local_train_dataloader = DataLoader(self.local_train_dataset, batch_size=8)
        self.delta = 1 / len(self.local_train_dataset)
    
    def trainer_init(self, tokenizer, accumulation_steps, batch_size, epochs, learning_rate, group_by_length, output_dir, 
        lora_rank: int = 16 , # Lora attention dimension 
        lora_alpha: int = 16, # The alpha parameter for Lora scaling
        lora_dropout: float = 0.05, # The dropout probability for Lora layers
        lora_module: List[str] = [ # The layers which need to be finetuned
            "q_proj",
        ],) -> None:
        # Use the transformer methods to perform the training steps
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, eps=1e-8)

        # Use differential privacy to ensure a DP algorithm where adding or removing a given element from the dataset, the answer 
        # from our algorithm will not change. This is done by adding Gaussian noise.
        self.model.train()
        self.model, optimizer, _ = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=self.local_train_dataloader,
            target_delta=self.delta,
            target_epsilon=7.5,
            epochs=epochs,
            max_grad_norm=MAX_GRAD_NORM,
        )

        # Initialize LoRA
        lora_config = LoraConfig(
            r=lora_rank, 
            lora_alpha=lora_alpha, 
            target_modules=lora_module, 
            lora_dropout=lora_dropout, 
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Get the PEFT model using LoRA
        self.model = get_peft_model(self.model._module, lora_config)
        
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
        
        self.local_trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.local_train_dataset,
            eval_dataset=self.local_eval_dataset,
            args=self.train_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            )
        )   

    def train(self) -> None:
        # Clear CUDA cache
        torch.cuda.empty_cache()
        self.local_trainer.train()
    
    def local_training(self) -> None:
        self.old_params = copy.deepcopy(
            OrderedDict(
                (name, param.detach()) for name, param in self.model.named_parameters() if "default" in name
            )
        )
        self.new_params = OrderedDict(
            (name, param.detach()) for name, param in self.model.named_parameters() if "default" in name
        )
        # self.model.state_dict = (
        #     lambda instance, *_, **__: get_peft_model_state_dict(
        #         instance, self.new_params, "default"
        #     )
        # ).__get__(self.model, type(self.model))
    
    def end_local_training(self, epoch, dataset_length, selected_clients, output_dir):
        dataset_length[self.client_id] = len(self.documents)
        new_weight = self.model._module.state_dict()
        output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(self.client_id))
        os.makedirs(output_dir, exist_ok=True)
        torch.save(new_weight, output_dir + "/pytorch_model.bin")

        old_weights = get_peft_model_state_dict(self.model._module, self.old_params, "default")
        set_peft_model_state_dict(self.model._module, old_weights, "default")
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