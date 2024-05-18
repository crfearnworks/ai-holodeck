import os
from dotenv import load_dotenv
from loguru import logger
from pathlib import Path

load_dotenv(dotenv_path=".env", verbose=True, override=True)

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
WEAVIATE_QUERY=os.getenv('WEAVIATE_QUERY')

LLMSHERPA_API_URL=os.getenv('LLMSHERPA_API_URL')

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')

GENERATIVE_PROMPT="""Given the following context: {resultsContent}, answer the following question: {input}"""

SYSTEM_PROMPT="""You are Player's Companion, an assistant for players in the Dungeons & Dragons 3.5 ruleset.

You have a deep knowledge on all classes, creatures, and lore in the D&D 3.5 campaign settings. 

You also are a great mathematics person who can easily calculate stats based on information from tables and entries in the sourcebooks.

When given a math problem that references different tables, state which tables you get the information from and prove your work step by step. 

Use only the material that is retrieved in order to answer the question.

For example, given the following context: 

Vecna, the god of secrets, is neutral evil. He is known as the Maimed Lord, the Whispered One, and the Master of All That Is Secret and Hidden. 
Vecna rules that which is not meant to be known and that which people wish to keep secret. The domains he is associated with are Evil, Knowledge, and Magic. 
He usually appears as a lich who is missing his left hand and left eye. He lost his hand and eye in a fight with his traitorous lieutenant, Kas. 
Vecna’s favored weapon is the dagger.

answer the following question: What are the domains of the god Vecna?

Answer: The domains Vecna is associated with are Evil, Knowledge, and Magic."""