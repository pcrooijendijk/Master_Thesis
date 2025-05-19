from DeepSeek import DeepSeekApplication, Metadata
from typing import Optional

sample_docs = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
]

client_id: int = 9
ori_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # The original model
lora_weights_path: str = "FL_output/pytorch_model.bin"        # Path to the weights after LoRA
lora_config_path: str = "FL_output"                           # Path to the config.json file after LoRA
prompt_template: str = 'utils/prompt_template.json'           # Prompt template for LLM
question: str = ""                                            # The question to be asked
uploaded_documents: Optional[str] = None                      # The corresponding document(s)
custom_text: Optional[str] = None                             # Custom text input instead of documents
temp: float = 0.1                                             # Temperature for token sampling
top_p: float = 0.75                                           # Top-p sampling
top_k: int = 40                                               # Top-k filtering
num_beams: int = 4                                            # Beam search size
max_new_tokens: int = 256

# Initalize a DeepSeek application for processing documents
deepseek = DeepSeekApplication(
    client_id, 
    ori_model,
    lora_weights_path,
    lora_config_path,
    prompt_template,
)

documents = []
metadata = {}

if sample_docs: 
    for index, doc in enumerate(sample_docs):
        with open(f"index_{index}.txt", "w") as file:
            file.write(doc)
            file.flush()
            content, metadata_doc, file_name = deepseek.doc_processor.process_file(f"index_{index}.txt")
            documents.append(content)
            metadata[file_name] = metadata_doc
print(f"documents test: {documents}")

# Load documents
deepseek.load_documents(documents, metadata)

# Query and retrieve the most relevant document
query = "Who introduced the theory of relativity?"
relevant_doc = deepseek.retrieve_relevant_docs(query, 10, 0.5)

# Generate an answer
answer = deepseek.generate_response(query, deepseek, top_k, top_p, num_beams, max_new_tokens, 0.28, temp, False)

print(f"Query: {query}")
print(f"Relevant Document: {relevant_doc}")
print(f"Answer: {answer}")

sample_queries = [
    "Who introduced the theory of relativity?",
    "Who was the first computer programmer?",
    "What did Isaac Newton contribute to science?",
    "Who won two Nobel Prizes for research on radioactivity?",
    "What is the theory of evolution by natural selection?"
]

expected_responses = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."
]

dataset = []

for query,reference in zip(sample_queries,expected_responses):

    relevant_docs = deepseek.retrieve_relevant_docs(query, 10, 0.5)
    response = deepseek.generate_response(query, deepseek, top_k, top_p, num_beams, max_new_tokens, 0.28, temp, False)
    print("RESPONSE", response)
    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts":relevant_docs,
            "response":response['content'],
            "reference":reference
        }
    )

from ragas import EvaluationDataset
evaluation_dataset = EvaluationDataset.from_list(dataset)

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper

evaluator_llm = LangchainLLMWrapper(llm)
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
print(result)