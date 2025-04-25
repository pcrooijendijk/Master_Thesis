# import gradio as gr

# def evaluate(question, files, pasted_text, temperature, top_p, top_k, beams, max_tokens):
#     # Simulated outputs
#     return (
#         "🧠 Model Output Here...",
#         "📁 Parsed Document Info Here...",
#         "📝 Output History Here..."
#     )

# def show_document():
#     # This function just returns visibility=True to reveal the box
#     return gr.update(visible=True, interactive=False)

# with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.blue)) as UI:
#     gr.Markdown("""
#     # 🔎 DeepSeek Q&A
#     ### Document Analysis and Question Answering.
#     Upload documents or paste text to ask questions about the content.
#     """)

#     with gr.Row():
#         # Left Column: Inputs
#         with gr.Column(scale=1):
#             question_input = gr.Textbox(
#                 lines=2,
#                 label="❓ Question",
#                 info="Upload documents below to ask questions about the content."
#             )

#             document_input = gr.MultimodalTextbox(
#                 file_count='multiple',
#                 placeholder="Upload your documents here.",
#                 label="📁 Document Input",
#                 show_label=True,
#                 info="Supported formats: PDF, DOCX, TXT"
#             )

#             pasted_text_input = gr.Textbox(
#                 lines=1,
#                 label="📃 Or paste text",
#                 info="Enter text directly. Each paragraph will be processed separately."
#             )

#             temperature_slider = gr.Slider(minimum=0, maximum=1, value=0.6, label="🌡️ Temperature")
#             top_p_slider = gr.Slider(minimum=0, maximum=1, value=0.75, label="Top p")
#             top_k_slider = gr.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k")
#             beams_slider = gr.Slider(minimum=1, maximum=4, step=1, value=4, label="Beams")
#             max_tokens_slider = gr.Slider(minimum=1, maximum=2000, step=1, value=1500, label="Max tokens")

#             generate_btn = gr.Button("🚀 Generate Response")

#         # Right Column: Outputs
#         with gr.Column(scale=1):
#             output_box = gr.Textbox(
#                 lines=10,
#                 label="🔮 Output",
#                 info="Output of the DeepSeek model.",
#                 interactive=False
#             )

#             metadata_box = gr.Textbox(
#                 lines=10,
#                 label="📊 Document Info",
#                 info="Meta Data of the input documents.",
#                 interactive=False
#             )

#             history_box = gr.Textbox(
#                 lines=20,
#                 label="📖 Output History",
#                 info="Questions and answers are displayed here.",
#                 interactive=False
#             )

#             # Button to show full document
#             show_doc_btn = gr.Button("📂 Show Full Document")

#             # Hidden textbox for full document content
#             full_doc_view = gr.Textbox(
#                 label="📄 Full Document Content",
#                 lines=20,
#                 visible=False,
#                 interactive=False
#             )

#     # Events
#     generate_btn.click(
#         fn=evaluate,
#         inputs=[
#             question_input,
#             document_input,
#             pasted_text_input,
#             temperature_slider,
#             top_p_slider,
#             top_k_slider,
#             beams_slider,
#             max_tokens_slider
#         ],
#         outputs=[output_box, metadata_box, history_box]
#     )

#     show_doc_btn.click(fn=show_document, outputs=full_doc_view)

# UI.queue().launch()

from langchain_core.documents import Document
doc, score = (Document(id='bc354813-fc6f-48d8-ac29-b195b911f9fd', metadata={'space_key_index': 1}, page_content='(2) academic re-searchers need equitable access to computationalresources; and (3) researchers should prioritize de-veloping efﬁcient models and hardware.2MethodsTo quantify the computational and environmen-tal cost of training deep neural network mod-els for NLP, we perform an analysis of the en-ergy required to train a variety of popular off-the-shelf NLP models, as well as a case study ofthe complete sum of resources required to developLISA (Strubell et al., 2018), a state-of-the-art NLPmodel from EMNLP 2018, including all tuningand experimentation.We measure energy use as follows. We train themodels described in §2.1 using the default settingsprovided, and sample GPU and CPU power con-sumption during training. Each model was trainedfor a maximum of 1 day. We train all models ona single NVIDIA Titan X GPU, with the excep-tion of ELMo which was trained on 3 NVIDIAGTX 1080 Ti GPUs. While training, we repeat-edly query the NVIDIA System Management In-terface2 to sample the GPU power'), 0.8155202865600586)
print(doc.page_content)