import pickle

with open('FL_output' + "/client_{}.pkl".format(1), "rb") as f:
    client = pickle.load(f)

print(client.get_client_id())
print(client.get_spaces())