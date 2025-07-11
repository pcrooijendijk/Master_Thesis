import gradio as gr
# from pdf2image import convert_from_path
import fire

from DeepSeek import DeepSeekApplication, Metadata

# For the output history
OUTPUT_HISTORY = []

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

    def format_qa_pair(pair: list[str]) -> str:
        formatted_str = ""
        for qa in pair:
            question, answer = qa
            formatted_str += f"Question: {question}\n\nAnswer: {answer}\n\n"
            formatted_str += "\n---------------------------------------------------------------------\n"
        return formatted_str


    def format_metadata_html(metadata: dict):
        flat_data = {}
        for k, v in metadata.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat_data[sub_k] = sub_v
            else:
                flat_data[k] = v

        html = "<ul style='list-style: none; padding-left: 0;'>"
        for k, v in flat_data.items():
            label = k.replace("_", " ").capitalize()
            value = v if v else "N/A"
            html += f"<li><strong>{label}:</strong> {value}</li>"
        html += "</ul>"
        return html

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

            deepseek.load_documents(documents, metadata[file_name])
            response, content_doc = deepseek.generate_response(question, deepseek, top_k, top_p, num_beams, max_new_tokens, 0.0, temp, True)
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
            deepseek.load_documents(documents, metadata, custom_text=True)
            response, content_doc = deepseek.generate_response(question, deepseek, top_k, top_p, num_beams, max_new_tokens, 0.0, temp, True)
        # If there are no documents uploaded, generate a prompt without extra context
        else:
            deepseek.load_documents([], metadata)
            response, content_doc = deepseek.generate_response(question, deepseek, top_k, top_p, num_beams, max_new_tokens, 0.28, temp, False)
        # For the output history
        OUTPUT_HISTORY.append([question, response['content']])
        return (response['content'], format_metadata_html(metadata), format_qa_pair(OUTPUT_HISTORY), content_doc) if uploaded_documents['files'] or custom_text else (response['content'], format_metadata_html(response['metadata']), format_qa_pair(OUTPUT_HISTORY), content_doc)

    def show_document():
        return gr.update(visible=True, interactive=False)
    
    with gr.Blocks(title="🔎 DeepSeek Q&A", theme=gr.themes.Default(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.blue)) as UI:
        gr.Markdown("""
        # 🔎 DeepSeek Document Q&A
        ### Document Analysis and Question Answering.
        Upload documents or paste text to ask questions about the content.
        """)

        with gr.Row():
            # Left Column: Inputs
            with gr.Column(scale=1):
                question_input = gr.Textbox(
                    lines=2,
                    label="❓ Question",
                    info="Upload documents below to ask questions about the content.", 
                    placeholder="What is this document about?"
                )

                document_input = gr.MultimodalTextbox(
                    file_count='multiple',
                    placeholder="Upload your documents here.",
                    label="📁 Document Input",
                    show_label=True,
                    info="Supported formats: PDF, DOCX, TXT"
                )

                pasted_text_input = gr.Textbox(
                    lines=1,
                    label="📃 Or paste text",
                    info="Enter text directly. Each paragraph will be processed separately.",
                    placeholder="Paste additional or alternative content here."
                )

                with gr.Accordion("⚙️ Advanced Generation Settings", open=False):
                    temperature_slider = gr.Slider(minimum=0, maximum=1, value=0.6, label="🌡️ Temperature (Randomness Control)")
                    top_p_slider = gr.Slider(minimum=0, maximum=1, value=0.75, label="Top p")
                    top_k_slider = gr.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k")
                    beams_slider = gr.Slider(minimum=1, maximum=4, step=1, value=4, label="Beams")
                    max_tokens_slider = gr.Slider(minimum=1, maximum=2000, step=1, value=1500, label="Max tokens")

                generate_btn = gr.Button("💬 Generate Response")

            # Right Column: Outputs
            with gr.Column(scale=1):
                output_box = gr.Textbox(
                    lines=10,
                    label="🔮 Output",
                    info="Output of the DeepSeek model.",
                    interactive=False
                )

                with gr.Accordion("📊 Document Info", open=False):
                    metadata_box = gr.HTML()

                history_box = gr.Textbox(
                    lines=10,
                    label="📖 Output History",
                    info="Questions and answers are displayed here.",
                    interactive=False
                )

                # Button to show full document
                show_doc_btn = gr.Button("📂 Show Full Document")

                # Hidden textbox for full document content
                full_doc_view = gr.Textbox(
                    label="📄 Full Document Content",
                    lines=20,
                    visible=False,
                    interactive=False
                )

        # Events
        generate_btn.click(
            fn=evaluate,
            inputs=[
                question_input,
                document_input,
                pasted_text_input,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                beams_slider,
                max_tokens_slider
            ],
            outputs=[output_box, metadata_box, history_box, full_doc_view]
        )

        show_doc_btn.click(fn=show_document, outputs=full_doc_view)

    UI.queue().launch(share=True)

if __name__ == "__main__":
    fire.Fire(run)
