import torch
import gradio as gr

from model import GPTModel, GPT_CONFIG_124M
from tokenizer_utils import tokenizer, text_to_token_ids, token_ids_to_text
from generate import generate

device = "cpu"

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("gpt_arxiv_best.pt", map_location=device))
model.eval()


def summarize_abstract(abstract):

    prompt = f"""
Summarize the following research abstract.

Abstract:
{abstract}

Summary:
"""

    input_ids = text_to_token_ids(prompt, device)

    token_ids = generate(
        model,
        input_ids,
        max_new_tokens=40,
        context_size=GPT_CONFIG_124M["context_length"],
        temperature=0.9,
        top_k=15
    )

    output_text = token_ids_to_text(token_ids)

    summary = output_text.split("Summary:")[-1].strip()

    return summary


demo = gr.Interface(
    fn=summarize_abstract,
    inputs=gr.Textbox(lines=12, label="Research Abstract"),
    outputs=gr.Textbox(label="Generated Summary"),
    title="Scientific Abstract Summarizer",
    description="GPT-2 model fine-tuned on arXiv abstracts."
)

demo.launch()