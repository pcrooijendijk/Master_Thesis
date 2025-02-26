# Testing how to get the parameters of ollama 
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-r1")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-r1")
prompt = "Explain quantum computing in simple terms."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))