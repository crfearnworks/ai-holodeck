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
        logger.info(f"Getting client with host: {host}")
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
            logger.info(f"{model} already exists on server...")
        else:
            logger.info(f"Pulling model {model}...")
            self.pull(model)
        return OllamaClient(model=model)
