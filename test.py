from fed_utils import Client, client_selection, Server
from utils import Dataset, Document, SpaceManagement, PromptHelper, Users
from datasets import load_dataset

global_model = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
local_model = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
output_dir = 'FL_output/'

documents = load_dataset("json", data_files="utils/documents.json")

# Intialize the spaces
space_names = ["mark", "new", "dev", "HR"]
space_keys = [0, 1, 2, 3]

users = Users(space_keys).get_users()

# Initialize the spaces, space manager, user accessor, user manager and the space permission manager by using a space management
management = SpaceManagement(space_names, space_keys, documents, users)