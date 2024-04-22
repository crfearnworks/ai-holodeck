from .client import get_embeddings_client, get_generative_client

gen_client = client.get_generative_client(host=constants.OLLAMA_LOCAL_URL)
embed_client = client.get_embeddings_client(host=constants.OLLAMA_LOCAL_URL)

    
