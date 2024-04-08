"""A Gradio client that can choose different LLMs for Q&A"""

import gradio as gr
from ollama import Client, pull as pullOllama

# Pull the model
pullOllama('mistral:7b')
pullOllama('llava:7b')
pullOllama('phi:2.7b')

# Set the tuple for all of the models
modelTuple = [
    ["Mistral","mistral:7b"],
    ["LLaVA","llava:7b"],
    ["Phi-2","phi:2.7b"]
]

def llmChat(model,text):
    
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
    llmChooser.launch(share=True)