import tenseal as ts
import torch
import os
import gc
from tqdm import tqdm
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
    
    def encrypt_model_weights(self, model_update, context, chunk_size=16384):
        encrypted_update = {}

        for k, v in tqdm(model_update.items(), desc="Encrypting"):
            tensor_flat = v.detach().cpu().numpy().flatten()
            encrypted_chunks = []

            for i in range(0, len(tensor_flat), chunk_size):
                chunk = tensor_flat[i : i + chunk_size]
                encrypted_chunk = ts.ckks_vector(context, chunk.tolist())
                encrypted_chunks.append(encrypted_chunk)

            encrypted_update[k] = encrypted_chunks

        return encrypted_update
    
    def decrypt_model_weights(self, encrypted_update, model_state_dict):
        decrypted_update = {}

        for name, encrypted_chunks in tqdm(encrypted_update.items(), desc="Decrypting"):
            # Decrypt and flatten each chunk
            flat_weights = []
            for chunk in encrypted_chunks:
                flat_weights.extend(chunk.decrypt())

            # Convert to tensor and reshape to the original shape
            original_shape = model_state_dict[name].shape
            decrypted_tensor = torch.tensor(flat_weights).view(original_shape)
            decrypted_update[name] = decrypted_tensor.to(model_state_dict[name].device)

        return decrypted_update


    def save_encrypted_weights(self, encrypted_weights, output_path, ouput_file: str="encrypted_weights.pkl"):
        output_dir = os.path.join(output_path, ouput_file)
        with open(output_dir, 'wb') as f:
            serialized = {}
            for k, v in encrypted_weights.items():
                # Wrap single CKKSVector in list for uniformity
                vectors = v if isinstance(v, list) else [v]
                serialized[k] = [chunk.serialize() for chunk in vectors]
            pickle.dump(serialized, f)