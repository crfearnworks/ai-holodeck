from weaviate.collections import Collection
import holodeck.ollama.embeddings as embeddings
import holodeck.ollama.ollama_client as ollama_client
import holodeck.weaviate.weaviate_utils as weaviate_utils
import holodeck.utilities.constants as constants
from loguru import logger
from unstructured.staging.base import convert_to_dict

def pipeline_prep(delete_collection: bool) -> None:
    
    logger.info("Getting embeddings client...")
    embeddingClient = ollama_client.OllamaClient.get_client(host=constants.OLLAMA_LOCAL_URL)
    logger.info("Setting up embedding model...")
    embeddingModel = ollama_client.OllamaClient.setup_model(embeddingClient, model=constants.DEFAULT_EMBEDDING_MODEL)    
    
    logger.info("Creating Weaviate client...")
    weaviateClient = weaviate_utils.create_weaviate_local_client()
    
    # adding this to delete basic chunking
    if delete_collection:
        logger.info("Deleting Weaviate collection...")
        weaviate_utils.delete_collection(weaviateClient, constants.WEAVIATE_COLLECTION_NAME)
    
    logger.info("Getting Weaviate collection...")
    weaviateCollection = weaviate_utils.get_collection(weaviateClient, constants.WEAVIATE_COLLECTION_NAME)
    
    elements = weaviate_utils.check_embedded_existance(weaviateClient, weaviateCollection, constants.EMBEDDED_DOCS_DIR, embeddingModel)
    if not elements:
        logger.info("No new elements to add to Weaviate collection")
    else:
        logger.info("Elements found to add to Weaviate collection")
        elementDictionary = convert_to_dict(elements=elements)
        elementChunks = class.generate_embeddings(embeddingModel, elementDictionary)
        weaviate_utils.load_chunks_into_weaviate(elementChunks, weaviateClient, weaviateCollection)
