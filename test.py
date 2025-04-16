import json 
client_selection_file = 'client_selection.json'

with open(client_selection_file, 'r') as openfile:
    selected_clients_index = json.load(openfile)
    selected_clients_index.pop()

with open(client_selection_file, 'w') as openfile:    
    json.dump(selected_clients_index, openfile)
