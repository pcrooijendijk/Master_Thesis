from langsmith import Client
from DeepSeek import DeepSeekApplication
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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


model_id = "meta-llama//Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

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

def llama_grader(inputs: dict, outputs: dict, reference_outputs: dict):
    question = inputs["question"]
    prediction = outputs["output"]
    reference = reference_outputs["output"]

    prompt = f"""
    [INST] <<SYS>>
    You are an expert evaluator. Please assess the quality of an AI answer.
    <</SYS>>

    Question: {question}

    Prediction: {prediction}

    Reference: {reference}

    Does the prediction correctly and fully answer the question based on the reference? Reply "yes" or "no", followed by a short explanation.
    [/INST]
    """

    result = pipe(prompt, max_new_tokens=150, return_full_text=False)[0]["generated_text"]

    return {
        "key": "correctness",
        "score": 1.0 if "yes" in result.lower() else 0.0,
        "value": result.strip()
    }

# After running the evaluation, a link will be provided to view the results in langsmith
experiment_results = client.evaluate(
    target,
    data="Sample dataset",
    evaluators=[
        llama_grader,
        # can add multiple evaluators here
    ],
    experiment_prefix="first-eval-in-langsmith",
    max_concurrency=2,
)