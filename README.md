# Master\_Thesis

Repository for my Master’s Thesis: **“Federated Learning for Secure and Permission-Aware LLM Applications”**

This project explores how Federated Learning (FL) can be combined with Large Language Models (LLMs) to enable secure, privacy-preserving, and permission-aware applications.

The system is divided into two main components: **training** and **inference**.

---

## 1. Training

Federated Learning is applied during the training stage.

* Main Script: `FL_main.py` which handles the FL process, including role and group assignment.
* Model: Training is demonstrated using DeepSeek, but the setup supports replacing it with any other LLM.
* Client–Server Setup:

  * `client_module.py` which defines the client-side training logic.
  * `Server.py` which aggregates client weights after training.
* Security:

  * Client weights are encrypted using Homomorphic Encryption (HE) before aggregation.
  * After aggregation, weights are decrypted to preserve security throughout the process.
* Dataset: Training is performed on the DOCBENCH dataset.

---

## 2. Inference

For inference, a user-friendly Gradio interface is provided:

* Retrieval-Augmented Generation (RAG): supports document retrieval for more accurate answers.
* Features:

  * Users can input questions directly.
  * View generated responses.
  * Inspect retrieved documents used for answering.

---

## Repository Structure

* `scores/output/`: Evaluation results, including `BLEU_scores`, `ixn_output`, and `RAGAS_scores`.
* `DeepSeek/`: DeepSeek implementation and post-processing of generated answers.
* `eval/`: Scripts for computing IXN, RAGAS, BLEU, and ROUGE scores, as well as test set selection.
* `fed_utils/`: Federated Learning utilities, including client and server modules.
* `img/`: Figures and plots generated from experimental results (used in the thesis).
* `perm_utils/`: Permission logic based on Confluence (roles, spaces, and permission management).
* `utils/`: Dataset preprocessing (`dataset.py`), JSON-formatted documents, the HE scheme, and LLM prompt templates.