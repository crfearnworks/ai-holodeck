import os
from dotenv import load_dotenv
from loguru import logger
from pathlib import Path

load_dotenv()

# system paths
LOGS_DIR=Path(os.getenv('LOGS_DIR'))
EMBEDDED_DOCS_DIR=Path(os.getenv('EMBEDDED_DOCS_DIR'))

# RAG pipeline models
DEFAULT_EMBEDDING_MODEL = os.getenv('DEFAULT_EMBEDDING_MODEL')
DEFAULT_GENERATOR_MODEL = os.getenv('DEFAULT_GENERATOR_MODEL')
OLLAMA_GEN_URL=os.getenv('OLLAMA_GEN_URL')
OLLAMA_EMBED_URL=os.getenv('OLLAMA_EMBED_URL')
OLLAMA_LOCAL_URL=os.getenv('OLLAMA_LOCAL_URL')

WEAVIATE_COLLECTION_NAME=os.getenv('WEAVIATE_COLLECTION_NAME')