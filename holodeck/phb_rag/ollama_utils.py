import ollama 
from holodeck.utilities import constants
from loguru import logger 
from typing import List, Dict
class OllamaClient(ollama.Client):
    def __init__(self, host = None, model = None):
        if host is None:
            host = constants.OLLAMA_LOCAL_URL
        super().__init__(host=host)
        self.model = model
        
    def __del__(self):
        self._client.close()

def get_generative_client(host = None) -> OllamaClient:
    if host is None:
        host = constants.OLLAMA_GEN_URL
    logger.info(f"Getting generative client with host: {host}")
    return OllamaClient(host=host)

def get_embeddings_client(host = None) -> OllamaClient:
    if host is None:
        host = constants.OLLAMA_EMBED_URL
    logger.info(f"Getting embeddings client with host: {host}")
    return OllamaClient(host=host)

def setup_embedding_model(client: OllamaClient, model_name = None) -> OllamaClient:
    if model_name is None:
        model_name = constants.DEFAULT_EMBEDDING_MODEL
    logger.info(f"Pulling embedding model {model_name}...")
    client.pull(model_name)
    return OllamaClient(model=model_name)

def setup_generative_model(client: OllamaClient, model_name = None) -> OllamaClient:
    if model_name is None:
        model_name = constants.DEFAULT_GENERATOR_MODEL
    logger.info(f"Pulling generative model {model_name}...")
    client.pull(model_name)
    return OllamaClient(model=model_name)

def generate_embeddings(client: OllamaClient, elements: List) -> List[Dict]:
    embeddings = []
    clientModel = client.model
    for element in elements:
        response = client.embeddings(model=clientModel, prompt=element["text"])
        embedding = response["embedding"]
        embeddings.append(embedding)

    chunk_embeddings_with_metadata = [
        {
            "id":  None,
            "type": element['type'],
            "title": element['metadata']['filename'],
            "url": "None",
            "content": element['text'],
            "label": "No Label",
            "tokens": len(element['text'].split()),
            "embedding": embedding,
        }
        for element, embedding in zip(elements, embeddings)
    ]
    return chunk_embeddings_with_metadata


def response_vectors(client: OllamaClient, query: str) -> List:
    response = client.embeddings(model=client.model, prompt=query)
    logger.debug(f"Response Vectors: {response}")
    return response["embedding"]

def generative_output(client: OllamaClient, query: str) -> List:
    response = client.generate(model=client.model, prompt=query)
    logger.debug(f"Generative Response: {response}")
    return response["response"]
    