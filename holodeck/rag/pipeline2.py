import os
from weaviate.collections import Collection
import holodeck.ollama.class as class
import holodeck.weaviate.weaviate_utils as weaviate_utils
import holodeck.utilities.constants as constants
from loguru import logger


def pipeline_prep() -> None:
    
    logger.info("Getting embeddings client...")
    embeddingClient = class.get_embeddings_client(constants.OLLAMA_LOCAL_URL)
    logger.info("Setting up embedding model...")
    embeddingModel = class.setup_embedding_model(embeddingClient, constants.DEFAULT_EMBEDDING_MODEL)    
    
    logger.info("Creating Weaviate client...")
    weaviateClient = weaviate_utils.create_weaviate_local_client()
    
    # adding this to delete basic chunking
    #logger.info("Deleting Weaviate collection...")
    #weaviate_utils.delete_collection(weaviateClient, constants.WEAVIATE_COLLECTION_NAME)
    
    logger.info("Getting Weaviate collection...")
    weaviateCollection = weaviate_utils.get_collection(weaviateClient, constants.WEAVIATE_COLLECTION_NAME)
    
    logger.info("Checking for documents in existing Weaviate collection...")
    file_path = constants.EMBEDDED_DOCS_DIR
    for filename in os.listdir(file_path):
        file = os.path.join(file_path,file)
        logger.info(f"Checking if {filename} exists in Weaviate")
        objectsResults = weaviateCollection.query.fetch_objects(return_properties=['title'])
        objects = [] 
        for obj in objectsResults.objects:
            objects.append(obj.properties['title'])
        logger.info(f"Objects in Weaviate: {objects}")
        directory = list(dict.fromkeys(objects))
        logger.info(f"Directory: {directory}")
        if filename not in directory:
            logger.info(f"{filename} does not exist in Weaviate. Adding file...")

            
        else:
            logger.info(f"{filename} exists in Weaviate.")
        