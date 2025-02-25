# The client should contain the following: name, documents assigned to this client, list of permissions and the local LLM

from utils import Permission
import transformers
from sklearn.model_selection import train_test_split
from DeepSeek import main as DeepSeek

from UserPermissionManagement import UserPermissionsResource

class Client:
    def __init__(self, client_id: int, name: str, user_permissions_resource: UserPermissionsResource, model) -> None:
        self.client_id = client_id
        self.name = name
        self.user_permissions_resource = user_permissions_resource
        self.permissions = set()
        self.model = model
        self.rest_user_permission_manager = user_permissions_resource.get_rest_user_permission_manager()
        self.space_manager = self.rest_user_permission_manager.get_space_manager()
        self.spaces = set()
        self.documents = []

        self.spaces_permissions_init()
        self.intialize_model()
    
    def spaces_permissions_init(self) -> None:
        permissions = self.user_permissions_resource.get_permissions(self.name, {"Username": self.name})
        for perm in permissions:
            # Add to the set of spaces this client has access to
            self.spaces.add((perm['spaceName'], perm['spaceKey']))
            for type in perm['permissions']:
                # Add to the set of permissions for this user
                self.permissions.add(type['permissionType']) if type['permissionGranted'] else None

    def filter_documents(self) -> None:
        for space in self.spaces:
            _, space_key = space
            # Only allow to add the documents to the Clients documents if the client has the permission
            # to view these documents
            if Permission.VIEWSPACE_PERMISSION.value in self.permissions:
                self.documents.append(self.space_manager.get_space(space_key).get_documents())
    
    def intialize_model(self) -> None:
        if self.model.lower() == "deepseek":
            DeepSeek()
        else: 
            print("Please indicate a valid model name.")

    def dataset_init(self, set_size) -> None:
        if set_size > 0:
            local_train = train_test_split(
                self.documents, test_size=0.7, shuffle=True, seed=42
            )
            self.local_train_dataset = local_train["train"]
            self.local_eval_dataset = local_train["test"]
    
    def trainer_init(self, tokenizer, accumulation_steps, batch_size, epochs, learning_rate, group_by_length):
        # Use the transformer methods to perform the training steps
        
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size, 
            gradient_accumulation_steps=accumulation_steps,
            warmup_steps=0,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            optim="adamw_torch",
            eval_steps=200,
            save_steps=200,
            output_dir="output",
            group_by_length=group_by_length,
            dataloader_drop_last=False
        )
        
        self.local_trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.local_train_dataset,
            eval_dataset=self.local_eval_dataset,
            args=self.train_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            )
        )
    
    def train(self):
        if not self.documents:
            print(f"Client {self.name} has no access to any documents to train the model on.")
            return self.model
        
        self.local_trainer.train()
    
    def send_update(self):
        # Encrypt the weights and send them to the global server
        pass

    def get_permissions(self):
        return self.permissions

    def get_spaces(self):
        return self.spaces