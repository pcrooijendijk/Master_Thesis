import json

class PromptHelper:
    def __init__(self, file_name: str):
        self.file_name = file_name

        self.loading_template()
    
    def loading_template(self) -> None:
        if not self.file_name:
            raise ValueError(f"Can't read {self.file_name}")
        
        with open(self.file_name) as file: 
            self.template = json.load(file)

    def generating_prompt(self, inst: str, input: str = None, label: str = None) -> str:
        if input: 
            prompt = self.template["prompt_input"].format(instruction=inst, input=input)
        else: 
            prompt = self.template["prompt_no_input"].format(instruction=inst)

        if label:
            prompt = f"{prompt}{label}"
        return  prompt

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()