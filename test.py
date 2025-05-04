def format_metadata_pretty(metadata: dict) -> list:
    flat_data = {}

    # Flatten first-level keys, especially for 'text_metadata'
    for k, v in metadata.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat_data[sub_k] = sub_v
        else:
            flat_data[k] = v

    # Make keys more human-readable
    pretty_keys = {
        "author": "Author",
        "creationDate": "Created",
        "creator": "Created With",
        "format": "Format",
        "keywords": "Keywords",
        "modDate": "Modified",
        "producer": "PDF Producer",
        "subject": "Subject",
        "title": "Title",
        "trapped": "Trapped",
        "processing_time": "Processing Time (s)",
        "context_length": "Context Length",
        "query_length": "Query Length"
    }

    # Format key-value pairs
    return [
        f"{pretty_keys.get(k, k)}: {v if v else 'N/A'}"
        for k, v in flat_data.items()
    ]

metadata = {'text_metadata': {'author': 'Andrey Kutuzov ; Lilja Øvrelid ; Terrence Szymanski ; Erik Velldal', 'creationDate': "D:20180609221548+02'00'", 'creator': 'TeX', 'format': 'PDF 1.3', 'keywords': '', 'modDate': "D:20180609221548+02'00'", 'producer': 'pdfTeX-1.40.16', 'subject': 'C18-1 2018', 'title': 'Diachronic word embeddings and semantic shifts: a survey', 'trapped': ''}, 'processing_time': 42.41308665275574, 'context_length': 2003, 'query_length': 75}
print(format_metadata_pretty(metadata))

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

import gradio as gr
with gr.Blocks() as demo:
    gr.Markdown("### Document Metadata")
    gr.HTML(format_metadata_html(metadata))

demo.launch()
