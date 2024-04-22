import ollama 
from holodeck.utilities import constants
from loguru import logger 
class OllamaClient(ollama.Client):
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

def setup_embed(client: OllamaClient, model_name: str) -> None:
    logger.info(f"Pulling embedding model {model_name}...")
    client.pull(model_name)

def setup_generate(client: OllamaClient, model_name: str) -> None:
    logger.info(f"Pulling generative model {model_name}...")
    client.pull(model_name)

