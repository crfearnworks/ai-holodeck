import holodeck.gradio.chatbot as chatbot


def main():
    
    chatbot.chat.launch(server_name="0.0.0.0", server_port=8000)


if __name__ == "__main__":
    main()