from fed_utils import Client
# import Server

from utils import Dataset, Instances, Role

# Loading the dataset
documents = Dataset("/media/sf_thesis/data_DocBench_test").get_documents()

# Intialize the spaces
space_names = ["mark", "new", "dev", "HR"]
space_keys = [0, 1, 2, 3]

users = {
    "admin": {
        "space": space_keys[0],
        "is_admin": True,
        "permissions": Role.get_role_permissions()["admin"]
    }, 
    "user1": {
        "space": space_keys[1],
        "is_admin": False,
        "permissions": Role.get_role_permissions()["editor"]
    }
}

inst = Instances(space_names, space_keys, documents, users)
user_permissions_resource = inst.get_user_permissions_resource()

# Get the permissions for user admin (target username, request)
print(user_permissions_resource.get_permissions('admin', {"Username": "admin"}))
print("----------------------------------------------------------")
print(user_permissions_resource.get_permissions('user1', {"Username": "user1"}))

# Define clients with different permissions -> Client(client_id, name, user_permissions_resource, model)
clients = [
    Client(client_id=1, name="admin", user_permissions_resource=user_permissions_resource, model="DeepSeek"),
    Client(client_id=2, name="user1", user_permissions_resource=user_permissions_resource, model="DeepSeek")
]

# # Initialize the server
# server = Server(num_clients=len(clients))

# # Run multiple federated learning rounds
# for round in range(3):
#     print(f"\n=== Federated Round {round + 1} ===")
#     server.run_federated_round(clients)
