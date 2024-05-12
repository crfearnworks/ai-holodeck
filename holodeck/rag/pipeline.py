from weaviate.collections import Collection
import holodeck.rag.ollama_utils as ollama_utils
import holodeck.rag.weaviate_utils as weaviate_utils
import holodeck.utilities.constants as constants
from loguru import logger
from unstructured.staging.base import convert_to_dict

def pipeline_prep() -> None:
    
    logger.info("Getting embeddings client...")
    embeddingClient = ollama_utils.get_embeddings_client(constants.OLLAMA_LOCAL_URL)
    logger.info("Setting up embedding model...")
    embeddingModel = ollama_utils.setup_embedding_model(embeddingClient, constants.DEFAULT_EMBEDDING_MODEL)    
    
    logger.info("Creating Weaviate client...")
    weaviateClient = weaviate_utils.create_weaviate_local_client()
    
    # adding this to delete basic chunking
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
        elementChunks = ollama_utils.generate_embeddings(embeddingModel, elementDictionary)
        weaviate_utils.load_chunks_into_weaviate(elementChunks, weaviateClient, weaviateCollection)
