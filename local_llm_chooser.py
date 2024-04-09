"""A Gradio client that can choose different LLMs for Q&A"""

import gradio as gr
import ollama
from ollama import Client, pull as pullOllama
from loguru import logger

# Set the tuple for all of the models
modelTuple = [
    ["Mistral","mistral:7b"],
    ["LLaVA","llava:7b"],
    ["Phi-2","phi:2.7b"]
]

def llmChat(model,text):
    try:
        logger.info(f"Selected {model}")
        ollama.show(model=model)
    except ollama.ResponseError as e:
        logger.info(f"Pulling {model}...")
        pullOllama(model)
    
    client = Client(host='http://localhost:11434')
    response = client.chat(model=model, messages=[
    {
        'role': 'user',
        'content': text,
    },
    ])
    return response['message']['content']

llmChooser = gr.Interface(
    llmChat,
    [
        gr.Dropdown(
            modelTuple, label="LLM", info="Choose an LLM from the dropdown."
        ),
        "text",
    ],
    "text"
)

if __name__ == '__main__':
    logger.info("Starting Gradio...")
    llmChooser.launch(share=True)