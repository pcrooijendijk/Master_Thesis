import json

class PromptHelper:
    def __init__(self, file_name: str):
        with open(file_name) as f: 
            self.template = json.load(f)
    
    def generate_prompt(self, question: str, document: str) -> str:
        response = self.template["prompt_input"].format(question=question, document=document)
        return response

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()