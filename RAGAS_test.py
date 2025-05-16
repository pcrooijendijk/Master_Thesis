from DeepSeek import DeepSeekApplication, Metadata

sample_docs = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
]

# Initialize RAG instance
client_id: int = 9,    
ori_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # The original model 
lora_weights_path: str = "FL_output/pytorch_model.bin", # Path to the weights after LoRA
lora_config_path: str = "FL_output", # Path to the config.json file after LoRA
prompt_template: str = 'utils/prompt_template.json', # Prompt template for LLM

question: str # The question to be asked
uploaded_documents: str = None, # The corresponding document(s)
custom_text: str = None, # User can give custum text as input instead of document(s)
temp: float = 0.1, # Temperature to module the next token probabilities
top_p: float = 0.75, # Only the smallest set of the most probable tokens with probabilities that add up to top_p or higher are kept for generation
top_k: int = 40, # Number of highest probability vocabulary tokens to keep for top-k-filtering
num_beams: int = 4, # Number of beams for beam search
max_new_tokens: int = 128,

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
    for doc in sample_docs:
        documents.append(doc)
        metadata['custom_input'] = Metadata(
            filename='custom_input',
            chunk_count=len(doc.split('\n')), 
            total_tokens=len(doc.split()),
            processing_time=0.0
        )

# Load documents
deepseek.load_documents(documents, metadata)

# Query and retrieve the most relevant document
query = "Who introduced the theory of relativity?"
relevant_doc = deepseek.retrieve_relevant_docs(query, 10)

# Generate an answer
answer = deepseek.generate_response(query, deepseek, top_k, top_p, num_beams, max_new_tokens, 0.28, temp, False)

print(f"Query: {query}")
print(f"Relevant Document: {relevant_doc}")
print(f"Answer: {answer}")