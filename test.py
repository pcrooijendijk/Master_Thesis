from utils import Users

# Intialize the spaces
space_names = ["mark", "new", "dev", "HR"]
space_keys = [0, 1, 2, 3]
users = Users(space_keys)
users.set_clients("lala", "model")
