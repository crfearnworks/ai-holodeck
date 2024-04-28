import gradio as gr
import holodeck.rag.pipeline as pipeline
import holodeck.rag.ollama_utils as ollama_utils
import holodeck.rag.weaviate_utils as weaviate_utils
import holodeck.utilities.constants as constants
import holodeck.utilities.custom_logging as custom_logging
import gradio as gr
from typing import List
from loguru import logger
from pprint import pprint
from unstructured.staging.base import convert_to_dict
from loguru import logger 

def pipeline(input: str) -> List:
    
    custom_logging.setup_logger()
    logger.info("Starting RAG pipeline...")
    
    logger.info("Getting generative client...")
    generativeClient = ollama_utils.get_generative_client(constants.OLLAMA_LOCAL_URL)
    logger.info("Setting up generative model...")
    generativeModel = ollama_utils.setup_generative_model(generativeClient)
    
    logger.info("Getting embeddings client...")
    embeddingClient = ollama_utils.get_embeddings_client(constants.OLLAMA_LOCAL_URL)
    logger.info("Setting up embedding model...")
    embeddingModel = ollama_utils.setup_embedding_model(embeddingClient)
    
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
        
    resultsContent = weaviate_utils.generate_results_content(weaviateClient, weaviateCollection, input)
    logger.info(f"results: {resultsContent}")
    generativePrompt = f"Using this data: {resultsContent}, respond to this prompt: {input}"
    logger.info(f"prompt: {generativePrompt}")
    response = ollama_utils.generative_output(generativeModel, generativePrompt)
    logger.info(f"response: {response}")
    return response
    
chat = gr.Interface(fn = pipeline, inputs="text", outputs="text")
    