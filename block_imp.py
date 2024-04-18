import gradio as gr
from ollama import Client

ollamaList = [
    ["Mistral","mistral:7b"],
    ["LLaVA","llava:7b"],
    ["Phi-2","phi:2.7b"]
]

with gr.Blocks() as server:
    with gr.Row():
        with gr.Column(scale=1):
            dropdown = gr.Dropdown(choices=ollamaList)
            top_k = gr.Slider(0.0, 100.0, label="top_k", value=40, info="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)")
            top_p = gr.Slider(0.0, 1.0, label="top_p", value=0.9, info=" Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)")
            temp = gr.Slider(0.0, 2.0, label="temperature", value=0.8, info="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)")
            num_refs = gr.Slider(0, 10, label="# of References", step=1, value=1, info="The number of references to retrieve from the collection. (Default: 1)")
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=590)
    with gr.Row():
        with gr.Column(scale=4):
            message = gr.Textbox(placeholder="Type here")
        with gr.Column(scale=1):
            submit = gr.Button("SEND")
            clear = gr.Button("CLEAR")
    
    def respond(message, chat_history):
        client = Client()
        bot_message = client.chat(model=dropdown, messages=[
            {
                'role': 'user',
                'content': message,
            },
        ])
        chat_history.append((message, bot_message))
        return "", chat_history

    message.submit(respond, [message, chatbot], [message, chatbot])



server.launch(server_name="0.0.0.0", server_port=8080, debug=True)