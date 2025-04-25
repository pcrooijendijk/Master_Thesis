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

import re

def clean_model_output(text: str) -> str:
    # Step 1: Remove special sentence tokens
    text = re.sub(r"<\｜begin▁of▁sentence\｜>", "", text)
    text = re.sub(r"<\｜end▁of▁sentence\｜>", "", text)

    # Step 2: Optional - Extract only answer after "Your answer"
    match = re.search(r"Your answer\s*(.*?)\s*$", text, re.DOTALL)
    if match:
        text = match.group(1)
    
    # Remove tags like </think>
    text = re.sub(r"</?\w+>", "", text)
    text = text.replace("\n", "")

    # Step 3: Strip leading/trailing whitespace
    return text.strip()


raw_output = """How to generically post process the following with regex: <｜begin▁of▁sentence｜>
        You are given a context document and a related question. Your task is to generate a comprehensive answer based on the context.

        Context:
        prior ﬁndings that it may provideplausible, but not faithful, explanation (Zhong et al.,2019). Interestingly, LIME does particularly wellacross these tasks in terms of faithfulness.From the ‘Random’ results that we concludemodels with overall poor performance on their ﬁ-nal tasks tend to have an overall poor ordering, withmarginal differences in comprehensiveness and suf-ﬁciency between them. For models that with highsufﬁciency scores: Movies, FEVER, CoS-E, and e-SNLI, we ﬁnd that random removal is particularlydamaging to performance, indicating poor absoluteranking; whereas those with high comprehensive-ness are sensitive to rationale length.7Conclusions and Future DirectionsWe have introduced a new publicly available re-source: the Evaluating Rationales And Simple En-glish Reasoning (ERASER) benchmark. This com-prises seven datasets, all of which include bothinstance level labels and corresponding supportingsnippets (‘rationales’) marked by human annotators.We have augmented many of any modelproperties. Examples include LIME (Ribeiro et al.,2016) and Alvarez-Melis and Jaakkola (2017);these methods approximate model behavior lo-cally by having it repeatedly make predictions overperturbed inputs and ﬁtting a simple, explainablemodel over the outputs.Acquiring rationales. Aside from interpretabilityconsiderations, collecting rationales from annota-tors may afford greater efﬁciency in terms of modelperformance realized given a ﬁxed amount of anno-tator effort (Zaidan and Eisner, 2008). In particular,recent work by McDonnell et al. (2017, 2016) hasobserved that at least for some tasks, asking anno-tators to provide rationales justifying their catego-rizations does not impose much additional effort.Combining rationale annotation with active learn-ing (Settles, 2012) is another promising direction(Wallace et al., 2010; Sharma et al., 2015).Learning from rationales. Work on learning fromrationales marked by annotators for text classiﬁca-tion dates back over a decade - Lim...

        Question:
        Are LIME and Alvarez-Melis and Jaakkola (2017) methods dependent on model properties?

        Instructions:
        - Answer based only on the given context if it's relevant.
        - If the context is insufficient or empty, provide the best answer using your own knowledge.
        - Based on the context above, explain your answer in complete sentences.
        - Ensure your answer is:
        1. Directly relevant
        2. Accurate and fact-based
        3. Complete and informative
        4. Clear and well-structured

        Please provide a full-sentence answer.
         Your answer
</think>

The context provided discusses various machine learning model explanations methods, such as LIME and Alvarez-Melis and Jaakkola (2017) methods. It mentions that these methods, like LIME, approximate the model's behavior locally by perturbing inputs and fitting a simple explanation model. It also notes that some methods, such as those by McDonnell et al. (2017, 2016), focus on providing rationales justifying categorizations, while others, like wallace et al. (2010) and Sharma et al. (2015), aim to learn from rationales. The key point is that these methods are dependent on the model's properties, such as locality and comprehensive-ness, rather than being independent of them.<｜end▁of▁sentence｜><｜end▁of▁sentence｜><｜end▁of▁sentence｜> """

cleaned = clean_model_output(raw_output)

print(cleaned)
