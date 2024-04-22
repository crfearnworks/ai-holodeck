from .client import get_embeddings_client, get_generative_client, setup_embed, setup_generate   
import holodeck.utilities.constants as constants
from loguru import logger

def pipeline_setup():
    
    logger.info("Setting up generative and embedding clients...")
    gen_client = get_generative_client(host=constants.OLLAMA_LOCAL_URL)
    embed_client = get_embeddings_client(host=constants.OLLAMA_LOCAL_URL)

    setup_embed(embed_client, constants.DEFAULT_EMBEDDING_MODEL)
    setup_generate(gen_client, constants.DEFAULT_GENERATOR_MODEL)

