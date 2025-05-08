from DeepSeek import DeepSeekApplication
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Optional, Dict
import ollama

all_documents = load_dataset("json", data_files="test_documents.json")
temp_doc = []

for index, _ in enumerate(all_documents['train']):
    temp_doc.append(all_documents["train"][index])

def _construct_prompt(query: str, answer: str, model_response: str) -> str:
    """Construct an enhanced prompt template"""
    return f"""
    Evaluate the quality of the AI-generated response below based on the given input and the correct output. 

    - **Input:** {query}  
    - **Expected Output:** {answer}  
    - **AI Response:** {model_response}  

    Score the AI response on a scale from 0 to 100, where 100 indicates a perfect match in accuracy, relevance, and completeness.
    """

def generate_response(
    prompt: str,
    context: Optional[List[str]] = None,
    max_context_length: int = 2000,
    temperature: int = 0.1,
    max_tokens: int = 2000,
) -> Dict:
    
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant that provides detailed, accurate answers based on the given context. If the context doesn\'t contain enough information to fully answer the question, acknowledge this and provide the best possible answer with the available information.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        options={
            'temperature': temperature,
            'max_tokens': max_tokens
        }
    )
    
    return {
        'content': response['message']['content'],
    }

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['question']}"
    )

    input_text = f"\n\n### Input:\n{entry['context']}" if entry["context"] else ""

    return instruction_text + input_text

deepseek = DeepSeekApplication(
    client_id = 9,    
    ori_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # The original model 
    lora_weights_path = "FL_output/pytorch_model.bin", # Path to the weights after LoRA
    lora_config_path = "FL_output", # Path to the config.json file after LoRA
    prompt_template = 'utils/prompt_template.json', # Prompt template for LLM
)

for entry in temp_doc:
    model_response = deepseek.test_generation(deepseek, entry['question'], entry['context'], 2000, 0.1, 10, 40, 4, 126)
    prompt = (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['answer']}`, "
        f"score the model response `{model_response}`"
        f" on a scale from 0 to 100, where 100 is the best score. "
    )
    print("\nPrompt")
    print(prompt)
    print("\nDataset response:")
    print(">>", entry['answer'])
    print("\nModel response:")
    print(">>", model_response)
    print("\nScore:")
    print(">>", generate_response(_construct_prompt(entry['question'], entry['answer'], model_response)))
    print("\n-------------------------")

def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['answer']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = generate_response(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores