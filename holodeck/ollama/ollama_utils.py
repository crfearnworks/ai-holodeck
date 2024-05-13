import ollama 
from holodeck.utilities import constants
from loguru import logger 
from typing import List, Dict


class OllamaClient(ollama.Client):
    host: str | None
    model: str | None
    
    def __init__(self, host = None, model = None):
        self.host = host
        self.model = model
        
    def __del__(self):
        self._client.close()

    def get_client(self, host):
        if host is None:
            self.host = constants.OLLAMA_LOCAL_URL
        logger.info(f"Getting generative client with host: {host}")
        return OllamaClient(host=host)
    
    def setup_model(self, model):
        model_list = self.list()
        # Flag to track if the host is found
        found = False
        # Loop through each model and check if host is found
        for model_name in model_list['models']:
            if model_name['model'] == model:
                found = True
                break
        if found:
            logger.info(f"Embedding model {model} already exists on server...")
        else:
            logger.info(f"Pulling embedding model {model}...")
            self.pull(model)
        return OllamaClient(model=model)

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
    
# Assume ollama is already imported and configured
def get_embedding(text):
    response = ollama.embeddings(
        model='mxbai-embed-large:v1',
        prompt=text
    )
    return response["embedding"]

def process_summary_items(data):
    embeddings = []
    for item in data:
        # Extracting text from the 'chunk' which is the second element of the tuple
        chunk_text = item['chunk'][1]
        summary_text = item['summary']

        # Generate embeddings
        chunk_embedding = get_embedding(chunk_text)
        summary_embedding = get_embedding(summary_text)

        # Store embeddings in a tuple or a dictionary as needed
        embeddings.append({
            'chunk_embedding': chunk_embedding,
            'summary_embedding': summary_embedding
        })
    return embeddings