import gradio as gr
import holodeck.rag.ollama_utils as ollama_utils
import holodeck.rag.weaviate_utils as weaviate_utils
import holodeck.utilities.constants as constants
import holodeck.utilities.custom_logging as custom_logging
from typing import List
from loguru import logger
from pprint import pprint
from unstructured.staging.base import convert_to_dict

# Set the tuple for all of the models
generativeTuple = [
    ["Mistral","mistral:7b"],
    ["LLaVA","llava:7b"],
    ["Phi-2","phi:2.7b"]
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
            embeddingDropdown = gr.Dropdown(embeddingTuple, label="Embedding Model", info="Choose a embedding model from the dropdown.", scale=1)
            input = gr.Textbox(placeholder="text", label="Input")
            input_btn = gr.Button("Submit")
        with gr.Column(scale=1):
            reference = gr.Textbox(label="Reference", placeholder="Reference Documents")
        with gr.Column(scale=2):
            output = gr.Textbox(label="Output", placeholder="Output will appear here")
            
    @input_btn.click(inputs=[generativeDropdown, embeddingDropdown, input], outputs=[output, reference])
    def pipeline(generativeDropdown: tuple, embeddingDropdown: tuple, input: str) -> List:
    
        custom_logging.setup_logger()
        logger.info("Starting RAG pipeline...")
        
        logger.info("Getting generative client...")
        generativeClient = ollama_utils.get_generative_client(constants.OLLAMA_LOCAL_URL)
        logger.info("Setting up generative model...")
        generativeModel = ollama_utils.setup_generative_model(generativeClient, generativeDropdown)
        
        logger.info("Getting embeddings client...")
        embeddingClient = ollama_utils.get_embeddings_client(constants.OLLAMA_LOCAL_URL)
        logger.info("Setting up embedding model...")
        embeddingModel = ollama_utils.setup_embedding_model(embeddingClient, embeddingDropdown)
        
        logger.info("Creating Weaviate client...")
        weaviateClient = weaviate_utils.create_weaviate_local_client()
        
        # adding this to delete basic chunking
        #logger.info("Deleting Weaviate collection...")
        #weaviate_utils.delete_collection(weaviateClient, constants.WEAVIATE_COLLECTION_NAME)
        
        logger.info("Getting Weaviate collection...")
        weaviateCollection = weaviate_utils.get_collection(weaviateClient, constants.WEAVIATE_COLLECTION_NAME)
        
        elements = weaviate_utils.check_embedded_existance(weaviateClient, weaviateCollection, constants.EMBEDDED_DOCS_DIR)
        logger.info(f"elements: {elements}")
        if not elements:
            logger.info("No new elements to add to Weaviate collection")
        else:
            logger.info("Elements found to add to Weaviate collection")
            for elem in elements:
                logger.info(f"Dictionary of {elem}: {elem.to_dict()}")
            elementDictionary = convert_to_dict(elements=elements)
            elementChunks = ollama_utils.generate_embeddings(embeddingModel, elementDictionary)
            weaviate_utils.load_chunks_into_weaviate(elementChunks, weaviateClient, weaviateCollection)
            
        resultsVectors = ollama_utils.response_vectors(embeddingModel, input)
        resultsContent, resultsReferences = weaviate_utils.generate_results_content(weaviateClient, weaviateCollection, input, resultsVectors)
        logger.info(f"results: {resultsContent}")
        logger.info(f"references: {resultsReferences}")
        generativePrompt = constants.GENERATIVE_PROMPT.format(input=input,resultsContent=resultsContent)
        logger.info(f"prompt: {generativePrompt}")
        response = ollama_utils.generative_output(generativeModel, generativePrompt)
        logger.info(f"response: {response}")
        return response, resultsReferences

chat.launch(server_name="0.0.0.0", server_port=8000)