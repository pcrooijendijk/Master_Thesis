class Server:
    def __init__(self, num_clients, global_model):
        self.global_model = global_model
        self.num_clients = num_clients
        
    def FedAvg(self, encrypted_updates, context):
        aggregated_update = {}

        for k in encrypted_updates[0].keys():
            # Get the number of chunks for this parameter
            num_chunks = len(encrypted_updates[0][k])

            # Initialize the chunk-wise sum
            chunk_sums = [encrypted_updates[0][k][i] for i in range(num_chunks)]

            # Accumulate each chunk from the remaining updates
            for update in encrypted_updates[1:]:
                for i in range(num_chunks):
                    chunk_sums[i] += update[k][i]

            # Average each chunk
            chunk_avg = [chunk * (1.0 / len(encrypted_updates)) for chunk in chunk_sums]

            aggregated_update[k] = chunk_avg

        return aggregated_update

    def get_server_context(self):
        return self.server_context