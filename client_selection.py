from fed_utils import client_selection
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, required=True)
    parser.add_argument("--frac_clients", type=float, required=True)
    args = parser.parse_args()

    # Selecting the indices of the clients which will be used for FL 
    selected_clients_index = client_selection(args.num_clients, args.frac_clients)
    selected_clients_index_list = [int(index) for index in selected_clients_index]

    # Safe the selected clients to json file 
    with open("client_selection.json", "w") as outputfile: 
        outputfile.write(json.dumps(list(selected_clients_index_list)))