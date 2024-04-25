import holodeck.phb_rag.pdf as pdf
import holodeck.phb_rag.ollama_utils as ollama_utils
import holodeck.phb_rag.weaviate_utils as weaviate_utils
import holodeck.gradio.chatbot as chatbot
import holodeck.utilities.constants as constants
import holodeck.utilities.custom_logging as custom_logging
from loguru import logger
from pprint import pprint

def main():
    
    custom_logging.setup_logger()
    logger.info("Starting PHB-RAG pipeline...")
    
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
    
    logger.info("Getting Weaviate collection...")
    weaviateCollection = weaviate_utils.get_collection(weaviateClient, constants.WEAVIATE_COLLECTION_NAME)
    
    elements = pdf.partition_pdf_elements_basic(constants.EMBEDDED_DOCS_DIR)
    
    elementDictionary = [element.to_dict() for element in elements]
    
    elementChunks = ollama_utils.generate_embeddings(embeddingModel, elementDictionary)
    
    weaviate_utils.load_chunks_into_weaviate(elementChunks, weaviateClient, weaviateCollection)
    
    logger.info("Querying Weaviate collection...")
    weaviateQuery = constants.WEAVIATE_QUERY
    resultsContent = []
    
    resultsVectors = ollama_utils.response_vectors(embeddingModel, weaviateQuery)
    
    results = weaviateCollection.query.near_vector(
        near_vector=resultsVectors["embedding"],
    )
    for obj in results.objects:
        resultsContent.append(obj.properties['content'])
    
    generativePrompt = f"Using this data: {resultsContent}, respond to this prompt: {weaviateQuery}"
    
    generativeOutput = generativeModel.generative_output(generativePrompt)
    
    pprint(generativeOutput)
    
if __name__ == "__main__":
    main()