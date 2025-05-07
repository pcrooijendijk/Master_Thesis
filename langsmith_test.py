from langsmith import Client
from DeepSeek import DeepSeekApplication
from ollama_res import OllamaResponder
import os

os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "qa-eval-master-thesis"

# Initalize a DeepSeek application for processing documents
deepseek = DeepSeekApplication(
    client_id = 9,    
    ori_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # The original model 
    lora_weights_path = "FL_output/pytorch_model.bin", # Path to the weights after LoRA
    lora_config_path = "FL_output", # Path to the config.json file after LoRA
    prompt_template = 'utils/prompt_template.json', # Prompt template for LLM
)

client = Client()

ollama_chat = OllamaResponder("llama3.2", 0.1, 126)

# Programmatically create a dataset in LangSmith
# For other dataset creation methods, see:
# https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_programmatically
# https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_in_application
dataset = client.create_dataset(
    dataset_name="Sample dataset", description="A sample dataset in LangSmith."
)

# Create examples
examples = [
    {
        "inputs": {"question": "Which country is Mount Kilimanjaro located in?"},
        "outputs": {"answer": "Mount Kilimanjaro is located in Tanzania."},
    },
    {
        "inputs": {"question": "What is Earth's lowest point?"},
        "outputs": {"answer": "Earth's lowest point is The Dead Sea."},
    },
]

# Add examples to the dataset
client.create_examples(dataset_id=dataset.id, examples=examples)
      
# Define the application logic you want to evaluate inside a target function
# The SDK will automatically send the inputs from the dataset to your target function
def target(inputs: dict) -> dict:
    result = ollama_chat.generate_response(inputs['question'])
    return {"answer": result["content"]}

# Define evaluator (can be OpenAI-based or another local LLM if supported)
def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt="Evaluate the factual correctness of the answer based on the input and reference.",
        model="openai:gpt-3.5-turbo",  # Use OpenAI, or switch to your own hosted model if supported
        feedback_key="correctness"
    )
    return evaluator(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)

# Run the evaluation
experiment_results = client.evaluate(
    target=target,
    data="Sample dataset",
    evaluators=[correctness_evaluator],
    experiment_prefix="ollama-local-eval",
    max_concurrency=2,
)