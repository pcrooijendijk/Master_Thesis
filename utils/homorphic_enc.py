import tenseal as ts
import torch
import os
import gc
import pickle

class HomomorphicEncryption:

    def __init__(self, context_dir="tenseal_context"):
        self.context_dir = context_dir
        os.makedirs(self.context_dir, exist_ok=True)
        self.context = None

    def generate_context(self):
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=32768,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            self.context.global_scale = 2**40
            self.context.generate_galois_keys()
            self.context.generate_relin_keys()

    def save_contexts(self):
        # Save private (full) context
        with open(os.path.join(self.context_dir, "full_context.tenseal"), "wb") as f:
            f.write(self.context.serialize(save_secret_key=True))

        # Save public context (no secret key)
        with open(os.path.join(self.context_dir, "public_context.tenseal"), "wb") as f:
            f.write(self.context.serialize(save_secret_key=False))

    def load_full_context(self):
        with open(os.path.join(self.context_dir, "full_context.tenseal"), "rb") as f:
            return ts.context_from(f.read())
    
    def load_public_context(self):
        with open(os.path.join(self.context_dir, "public_context.tenseal"), "rb") as f:
            return ts.context_from(f.read())
    
    def load_encrypted_weights(self, serialized):
        context = self.load_full_context()
        
        encrypted_weights = {}
        for k, chunks_serialized in serialized.items():
            chunks = [ts.ckks_vector_from(context, chunk_bytes) for chunk_bytes in chunks_serialized]
            encrypted_weights[k] = chunks

        return encrypted_weights
        
    def encrypt_model_weights(self, state_dict, context, chunk_size=32768//2):
        encrypted_layers = {}

        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                tensor = param.detach().cpu().numpy().flatten().tolist()

                if len(tensor) <= chunk_size:
                    # Can fit into one ciphertext
                    encrypted_layers[name] = [ts.ckks_vector(context, tensor)]
                else:
                    # Split into multiple chunks
                    encrypted_chunks = []
                    for i in range(0, len(tensor), chunk_size):
                        chunk = tensor[i:i + chunk_size]
                        encrypted_chunks.append(ts.ckks_vector(context, chunk))
                    encrypted_layers[name] = encrypted_chunks

        return encrypted_layers   
    
    def decrypt_model_weights(self, model, encrypted_aggregated):
        decrypted_state = {}
        context = self.load_full_context()

        for name, encrypted_chunks in encrypted_aggregated.items():
            flat_weights = []

            # Handle multiple chunks per parameter
            for chunk in encrypted_chunks:
                flat_weights.extend(chunk.decrypt())

            # Reshape to original tensor shape
            original_shape = model.state_dict()[name].shape
            decrypted_tensor = torch.tensor(flat_weights).view(original_shape)
            decrypted_state[name] = decrypted_tensor
            del flat_weights
            gc.collect()

        return decrypted_state

    def save_encrypted_weights(self, encrypted_weights, output_path, ouput_file: str="encrypted_weights.pkl"):
        output_dir = output_path + "/" + ouput_file
        with open(output_dir, 'wb') as f:
            # Serialize each layer and store in a dict of bytes
            serialized = {
                k: [chunk.serialize() for chunk in v]  # v is a list of ckks_vector chunks
                for k, v in encrypted_weights.items()
            }
            # Use pickle to write the entire dict
            pickle.dump(serialized, f)