import torch
import gradio as gr
import fire

from utils.prompt_template import PromptHelper
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig

from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(
    ori_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # The original model 
    lora_weights_path: str = "FL_output/pytorch_model.bin", # Path to the weights after LoRA
    # lora_config_path: str = "FL_output", # Path to the config.json file after LoRA
    prompt_template: str = 'utils/prompt_template.json', # Prompt template for LLM
):
    prompter = PromptHelper(prompt_template)
    config = AutoConfig.from_pretrained(ori_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ori_model,
        config=config,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)
    lora_weights = torch.load(lora_weights_path)
    model = PeftModel(model, config)
    set_peft_model_state_dict(model, lora_weights, "default")

    tokenizer = AutoTokenizer.from_pretrained(ori_model)    
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"

    model.half()
    model.eval() # Set the model to evaluation mode

    def evaluate(
        question: str, # The question to be asked
        document: str = None, # The corresponding document(s)
        temp: float = 0.1, # Temperature to module the next token probabilities
        top_p: float = 0.75, # Only the smallest set of the most probable tokens with probabilities that add up to top_p or higher are kept for generation
        top_k: int = 40, # Number of highest probability vocabulary tokens to keep for top-k-filtering
        num_beams: int = 4, # Number of beams for beam search
        max_new_tokens: int = 128
    ):
        prompt_input = prompter.generate_prompt(question, document)
        inputs = tokenizer(prompt_input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        config_gen = GenerationConfig(
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=config_gen,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        seq = generation_output.sequences[0]
        output = tokenizer.decode(seq)
        yield prompter.get_response(output)

    UI = gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Question",
                placeholder="Tell me about alpacas.",
            ),
            gr.components.Textbox(lines=2, label="Document", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output"),
        ],
        outputs=[
            gr.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="FederatedGPT-shepherd",
        description="Shepherd is a LLM that has been fine-tuned in a federated manner ",
    ).queue()

    UI.launch(share=True, server_port=7860)

if __name__ == "__main__":
    fire.Fire(run)