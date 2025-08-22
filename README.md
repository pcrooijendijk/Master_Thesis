# Master_Thesis
Repository for my Master Thesis: " Federated Learning for Secure and Permission-Aware LLM Applications"

The whole projects consists of two main methods: training and inference.
1. Training: Federated Learning is applied when training. `FL_main.py` is used for this training process where there are roles and groups specified. The training is done using DeepSeek, but any other model can be used instead. The server and clients are made for the FL setup and the DOCBENCH dataset is used for this process. Training is done within the `client_module.py` file. The `Server.py` module is used for aggregating the weights of the clients after training. After aggregating, the weights are decrypted using Homomorphic Encryption, where the encrypting process took place before aggregating the weights. 


2. Inference: The Gradio library is used for the interface where RAG is used for document retrieval. This interface enables users to easily input questions, view responses from the model, and inspect the full documents retrieved for question answering.

Overview of the other files: 
- Any directory ending with `scores/output` are the scores used for evaluating the framework (`BLEU_scores`, `ixn_output`, `RAGAS_scores`). 
- `DeepSeek` contains the DeepSeek application and implemenation. Most importantly, it holds the post processing of the answers after generation.
- `eval` holds the files for computing the IXN, RAGAS, and BLEU/ROUGE scores. It also contains the file for test set selection.
- `fed_utils` contains the aforementioned client and server modules.
- `img` contains all the figures used in the thesis generated from the experimental results.
- `perm_utils` holds the logic for the permissions based on Confluence, such as the permissions, roles, spaces and permissions management.
- `utils` contains the preprocessing of the dataset in `dataset.py`, the documents in JSON format, the Homomorphic Encryption scheme, and the prompt template for the LLM.