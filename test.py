import gradio as gr

def evaluate(
    question: str, # The question to be asked
    document: str = None, # The corresponding document(s)
    pasted_text: str = None,
    temp: float = 0.1, # Temperature to module the next token probabilities
    top_p: float = 0.75, # Only the smallest set of the most probable tokens with probabilities that add up to top_p or higher are kept for generation
    top_k: int = 40, # Number of highest probability vocabulary tokens to keep for top-k-filtering
    num_beams: int = 4, # Number of beams for beam search
    max_new_tokens: int = 128,
    **kwargs,
):
    return "test test", document

UI = gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Textbox(
            lines=2,
            label="❓Question",
            info="Upload documents below to ask questions about the content.",
        ),
        gr.MultimodalTextbox(
            file_types=['txt', 'pdf', 'docx'],
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
            minimum=1, maximum=2000, step=1, value=500, label="Max tokens"
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