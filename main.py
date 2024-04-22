import holodeck.phb_rag.pipeline as pipeline
import holodeck.phb_rag.client as client
import holodeck.utilities.constants as constants
import holodeck.utilities.custom_logging as custom_logging

def main():
    
    custom_logging.setup_logger()
    
    client.setup_embed(embed_client, constants.DEFAULT_EMBEDDING_MODEL)
    client.setup_generate(gen_client, constants.DEFAULT_GENERATOR_MODEL)
    
    
    
if __name__ == "__main__":
    main()