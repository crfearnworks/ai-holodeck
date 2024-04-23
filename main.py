import holodeck.phb_rag.pdf as pdf
import holodeck.phb_rag.ollama_utils as ollama_utils
import holodeck.phb_rag.weaviate_utils as weaviate_utils
import holodeck.utilities.constants as constants
import holodeck.utilities.custom_logging as custom_logging
from pprint import pprint

def main():
    
    custom_logging.setup_logger()
    
    generativeClient = ollama_utils.get_generative_client(constants.OLLAMA_LOCAL_URL)
    generativeModel = ollama_utils.setup_generative_model(generativeClient)
    
    embeddingClient = ollama_utils.get_embeddings_client(constants.OLLAMA_LOCAL_URL)
    embeddingModel = ollama_utils.setup_embedding_model(embeddingClient)
    
    weaviateClient = weaviate_utils.create_weaviate_local_client()
    
    collection = weaviate_utils.main(weaviateClient, "CMMC")
    
    CMMCQuery = "What is the family for Access Control?"
    
    resultsContent = []
    results = collection.query.near_vector(
        near_vector=embeddingModel.response_vectors(CMMCQuery),
    )
    for obj in results.objects:
        resultsContent.append(obj.properties['content'])
    
    generativePrompt = f"Using this data: {resultsContent}, respond to this prompt: {CMMCQuery}"
    
    generativeOutput = generativeModel.generative_output(generativePrompt)
    
    pprint(generativeOutput)
    
if __name__ == "__main__":
    main()