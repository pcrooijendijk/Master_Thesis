from langsmith import Client
from DeepSeek import DeepSeekApplication
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
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
    return deepseek.test_generation(deepseek, inputs["question"], 0.1, 0.75, 40, 4, 128)

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="openai:o3-mini",
        feedback_key="correctness",
    )
    eval_result = evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs
    )
    return eval_result

# After running the evaluation, a link will be provided to view the results in langsmith
experiment_results = client.evaluate(
    target,
    data="Sample dataset",
    evaluators=[
        correctness_evaluator,
        # can add multiple evaluators here
    ],
    experiment_prefix="first-eval-in-langsmith",
    max_concurrency=2,
)