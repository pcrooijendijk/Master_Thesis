import ollama
from typing import List, Optional, Dict

class OllamaResponder():

    def __init__(self, model_name: str, temperature: int, max_tokens: int):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate_response(
        self,
        prompt: str,
    ) -> Dict:
        
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': 'Evaluate the factual correctness of the answer based on the input and reference.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }
        )
        
        return {
            'content': response['message']['content']
        }
