import gradio as gr
from holodeck.ollama.ollama_client import OllamaClient
import holodeck.weaviate.weaviate_utils as weaviate_utils
import holodeck.utilities.constants as constants
import holodeck.utilities.custom_logging as custom_logging
import holodeck.rag.pipeline as pipeline_prep
import holodeck.ollama.embeddings as embeddings
from IPython.display import display, HTML
from typing import List
from loguru import logger

# Set the tuple for all of the models
generativeTuple = [
    ["mistral:7b-instruct-v0.2-q8_0","mistral:7b-instruct-v0.2-q8_0"],
    ["llava:7b-v1.6-mistral-q8_0","llava:7b-v1.6-mistral-q8_0"],
    ["llama3:8b-instruct-q8_0","llama3:8b-instruct-q8_0"],
    ["phi3:3.8b-mini-instruct-4k-q4_K_M","phi3:3.8b-mini-instruct-4k-q4_K_M"],
    ["phi3:3.8b-mini-instruct-4k-fp16","phi3:3.8b-mini-instruct-4k-fp16"],
    ["llava-phi3:3.8b","llava-phi3:3.8b"],
    ["llava-llama3:8b-v1.1-q4_0","llava-llama3:8b-v1.1-q4_0"],
    ["llava-llama3:8b-v1.1-fp16","llava-llama3:8b-v1.1-fp16"],
    ["command-r:35b-v0.1-q4_0","command-r:35b-v0.1-q4_0"]
]

embeddingTuple = [
    ["mxbai-embed-large","mxbai-embed-large:v1"],
    ["nomic-embed-text","nomic-embed-text:v1.5"],
    ["all-minilm","all-minilm:l6"]
]

with gr.Blocks() as chat:
    with gr.Row():
        with gr.Column(scale=1):
            generativeDropdown = gr.Dropdown(generativeTuple, label="Generative Model", info="Choose an generative model from the dropdown.", scale=1)
            input = gr.Textbox(placeholder="text", label="Input")
            input_btn = gr.Button("Submit")
        with gr.Column(scale=1):
            reference = gr.Textbox(label="Reference", placeholder="Reference Documents")
        with gr.Column(scale=2):
            context = gr.Textbox(label="Context", placeholder="Chunks referenced will appear here")
    with gr.Row():
        output = gr.Textbox(label="Output", placeholder="Output will appear here")
            
    @input_btn.click(inputs=[generativeDropdown, input], outputs=[output, reference, context])
    def pipeline(generativeDropdown: tuple, input: str) -> List:
    
        custom_logging.setup_logger()
        logger.info("Starting RAG pipeline...")
        
        logger.info("Getting generative client...")
        generativeClient = OllamaClient.get_client(constants.OLLAMA_LOCAL_URL)
        logger.info("Setting up generative model...")
        generativeModel = OllamaClient.setup_model(generativeClient, generativeDropdown)
        
        logger.info("Getting embeddings client...")
        embeddingClient = OllamaClient.get_client(constants.OLLAMA_LOCAL_URL)
        logger.info("Setting up embedding model...")
        embeddingModel = OllamaClient.setup_model(embeddingClient, constants.DEFAULT_EMBEDDING_MODEL)  
        
        logger.info("Creating Weaviate client...")
        weaviateClient = weaviate_utils.create_weaviate_local_client()
        
        logger.info("Getting Weaviate collection...")
        weaviateCollection = weaviate_utils.get_collection(weaviateClient, constants.WEAVIATE_COLLECTION_NAME)
        
        resultsVectors = embeddings.response_vectors(embeddingModel, input)
        resultsContent, resultsReferences = weaviate_utils.generate_results_content(weaviateClient, weaviateCollection, input, resultsVectors)
        logger.info(f"results: {resultsContent}")
        logger.info(f"references: {resultsReferences}")
        generativePrompt = constants.GENERATIVE_PROMPT.format(input=input,resultsContent=resultsContent)
        logger.info(f"prompt: {generativePrompt}")
        response = embeddings.generative_output(generativeModel, generativePrompt)
        logger.info(f"response: {response}")
        return response, resultsReferences, resultsContent
    
    chat.load(pipeline_prep.pipeline_prep(delete_collection=False))

chat.launch(server_name="0.0.0.0", server_port=8000)