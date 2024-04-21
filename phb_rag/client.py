import ollama 
from constants import OLLAMA_EMBED_URL, OLLAMA_GEN_URL
from loguru import logger 
class OllamaClient(ollama.Client):
    def __del__(self):
        self._client.close()

def get_generative_client(host = None) -> OllamaClient:
    if host is None:
        host = OLLAMA_GEN_URL
    logger.info(f"Getting generative client with host: {host}")
    return OllamaClient(host=host)

def get_embeddings_client(host = None) -> OllamaClient:
    if host is None:
        host = OLLAMA_EMBED_URL
    logger.info(f"Getting embeddings client with host: {host}")
    return OllamaClient(host=host)

def setup_embed(client: OllamaClient, model_name: str) -> None:
    client.pull(model_name)

def setup_generate(client: OllamaClient, model_name: str) -> None:
    client.pull(model_name)