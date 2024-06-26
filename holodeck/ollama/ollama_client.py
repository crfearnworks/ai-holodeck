import ollama 
from holodeck.utilities import constants
from loguru import logger 

class OllamaClient(ollama.Client):
    def __init__(self, host = None, model = None):
        super().__init__()
        self.host = host
        self.model = model
        logger.debug(f"Initialized OllamaClient with host: {self.host} and model: {self.model}")
        
    def close(self):
        """Close the client connection."""
        self._client.close()

    @classmethod
    def get_client(cls, host=None):
        if host is None:
            logger.debug("Host is None, setting to default.")
            host = constants.OLLAMA_LOCAL_URL
        logger.info(f"Getting client with host: {host}")
        return cls(host=host)
    
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
