import gradio as gr
# from pdf2image import convert_from_path
import fire

from DeepSeek import DeepSeekApplication, Metadata

def run(
    client_id: int = 9,    
    ori_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # The original model 
    lora_weights_path: str = "FL_output/pytorch_model.bin", # Path to the weights after LoRA
    lora_config_path: str = "FL_output", # Path to the config.json file after LoRA
    prompt_template: str = 'utils/prompt_template.json', # Prompt template for LLM
):
    
    # Initalize a DeepSeek application for processing documents
    deepseek = DeepSeekApplication(
        client_id, 
        ori_model,
        lora_weights_path,
        lora_config_path,
        prompt_template,
    )

    def evaluate(
        question: str, # The question to be asked
        uploaded_documents: str = None, # The corresponding document(s)
	    custom_text: str = None, # User can give custum text as input instead of document(s)
        temp: float = 0.1, # Temperature to module the next token probabilities
        top_p: float = 0.75, # Only the smallest set of the most probable tokens with probabilities that add up to top_p or higher are kept for generation
        top_k: int = 40, # Number of highest probability vocabulary tokens to keep for top-k-filtering
        num_beams: int = 4, # Number of beams for beam search
        max_new_tokens: int = 128,
    ):  
        documents = []
        metadata = {}

        # If there are documents uploaded, then the documents are processed and used for generating the prompt
        if uploaded_documents['files']:
            for file in uploaded_documents['files']: 
                content, metadata_doc, file_name = deepseek.doc_processor.process_file(file)
                documents.append(content)
                metadata[file_name] = metadata_doc

            deepseek.load_documents(documents, metadata)
            response = deepseek.generate_response(question, deepseek, top_k, top_p, num_beams, max_new_tokens, 0.0, temp, True)
        # If there is manual input from the user, treat it as if it is a document and append to the total list of documents
        elif custom_text:  
            content_custom = custom_text.strip()
            if content_custom: 
                documents.append(content_custom)
                metadata['custom_input'] = Metadata(
                    filename='custom_input',
                    chunk_count=len(content_custom.split('\n')), 
                    total_tokens=len(content_custom.split()),
                    processing_time=0.0
                )
            deepseek.load_documents(documents, metadata)
            response = deepseek.generate_response(question, deepseek, top_k, top_p, num_beams, max_new_tokens, 0.0, temp, True)
        # If there are no documents uploaded, generate a prompt without extra context
        else:
            deepseek.load_documents([], metadata)
            response = deepseek.generate_response(question, deepseek, top_k, top_p, num_beams, max_new_tokens, 0.28, temp, False)
        return (response['content'], metadata) if uploaded_documents['files'] or custom_text else (response['content'], response['metadata'])

    # The Gradio interface for fetching the question, documents, custom input and parameters
    UI = gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="❓Question",
                info="Upload documents below to ask questions about the content.",
            ),
            gr.MultimodalTextbox(
                file_count='multiple',
                placeholder="Upload your documents here.",
                label="📁 Document Input",
                show_label=True,
                info="Supported formats: PDF, DOCX, TXT"
            ),
            gr.components.Textbox(
                lines=1,
                label="📃 Or paste text",
                info="Enter text directly. Each paragraph will be processed seperately."
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.6, label="🌡️ Temperature"
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
                minimum=1, maximum=2000, step=1, value=1500, label="Max tokens"
            ),
        ],
        outputs=[
            gr.Textbox(
                lines=10,
                label="🔮 Output",
                info="Output of the DeepSeek model."
            ),
            gr.Textbox(
                lines=10, 
                label="📊 Document Info",
                info="Meta Data of the input documents."
            )
        ],
        title="🔎 DeepSeek Q&A",
        description=""" 
            ### Document Analysis and Question Answering.
            # Upload documents or paste text to ask questions about the content.
        """ ,
        theme=gr.themes.Default(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.blue),
        submit_btn="Generate Response",
        flagging_mode="never"
    ).queue()

    UI.launch(share=True, server_port=7860)

if __name__ == "__main__":
    fire.Fire(run)