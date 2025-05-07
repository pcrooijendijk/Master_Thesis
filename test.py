from utils import Dataset
from DeepSeek import DeepSeekApplication
from datasets import load_dataset
import urllib.request
import json 
from tqdm import tqdm

dataset = Dataset("data_DocBench_test")
dataset.convert_to_json(1, "test_documents.json")

all_documents = load_dataset("json", data_files="utils/documents.json")
temp_doc = []
for index, _ in enumerate(all_documents['train']):
    temp_doc.append(all_documents["train"][index])

def query_model(
    prompt,
    model="llama3",
    url="http://localhost:11434/api/chat"
):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

deepseek = DeepSeekApplication(
    client_id = 9,    
    ori_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # The original model 
    lora_weights_path = "FL_output/pytorch_model.bin", # Path to the weights after LoRA
    lora_config_path = "FL_output", # Path to the config.json file after LoRA
    prompt_template = 'utils/prompt_template.json', # Prompt template for LLM
)

for entry in temp_doc:
    model_response = deepseek.test_generation(deepseek, entry['question'], 0.1, 10, 40, 4, 126)
    prompt = (
        f"Given the input `{format_input(entry['question'])}` "
        f"and correct output `{entry['answer']}`, "
        f"score the model response `{model_response}`"
        f" on a scale from 0 to 100, where 100 is the best score. "
    )
    print("\nDataset response:")
    print(">>", entry['output'])
    print("\nModel response:")
    print(">>", entry["model_response"])
    print("\nScore:")
    print(">>", query_model(prompt))
    print("\n-------------------------")

def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores