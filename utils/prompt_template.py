import json

class PromptHelper:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.base_template = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
            The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
            The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.

            User: {question}
            Assistant: {think_tag}
            {reasoning}
            {think_end_tag}
            {answer_tag}
            {solution}
            {answer_end_tag}
        """

        self.loading_template()
    
    def loading_template(self) -> None:
        if not self.file_name:
            raise ValueError(f"Can't read {self.file_name}")
        
        with open(self.file_name) as file: 
            self.template = json.load(file)

        # def _construct_prompt(self, query: str, context: str) -> str:
        # """Construct an enhanced prompt template"""
        # return f"""
        # Context Information:
        # {context}

        # Question: {query}

        # Please provide a comprehensive answer based on the context above. Consider:
        # 1. Direct relevance to the question
        # 2. Accuracy of information
        # 3. Completeness of response
        # 4. Clarity and coherence

        # Answer:
        # """

    def generating_prompt(self, question: str, document: str = None, answer: str = None) -> str:
        return self.base_template.format(
            question="""You are a helpful assistant that helps users answer questions based on the given document. 
            You should answer the following question about the given document: {}
            Answer the question based on the given text:""".format(question),
            think_tag="<think>",
            reasoning="""1. Understand what the document is about.
            2. Break down the different sections of the document.
            3. Apply natural language processing and information retrieval.
            4. Give answer to the question. """,
            think_end_tag="<think>",
            answer_tag="<answer>",
            solution="[Answer to the question will be provided here]",
            answer_end_tag="<answer>"
        )

    # def generating_prompt(self, inst: str, input: str = None, label: str = None) -> str:
    #     if input: 
    #         prompt = self.template["prompt_input"].format(instruction=inst, input=input)
    #     else: 
    #         prompt = self.template["prompt_no_input"].format(instruction=inst)

    #     if label:
    #         prompt = f"{prompt}{label}"
    #     return prompt

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()