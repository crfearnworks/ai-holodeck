import weaviate
import weaviate.classes as wvc
from weaviate.collections import Collection
from weaviate.client import WeaviateClient
from pdf import partition_pdf_elements_basic
import holodeck.utilities.constants as constants 
from typing import List, Dict
from loguru import logger

class WeaviateClient(weaviate):
    def __del__(self):
        self._client.close()

def create_weaviate_local_client() -> WeaviateClient:
    client = weaviate.connect_to_local(
        additional_config=wvc.init.AdditionalConfig(
            timeout=(60,1800),
        ),
    )
    return client

def create_collection(client: WeaviateClient, collection_name: str)-> Collection:
    with client: 
        client.collections.delete(collection_name)
        client.collections.create(
            name=collection_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(),
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE # select prefered distance metric
        ),
    )
    collection = client.collections.get(name=collection_name)
    return collection

def load_chunks_into_weaviate(chunks: List[Dict], client: WeaviateClient, collection_name: str):
    collection = create_collection(client, collection_name)
    chunk_objs = []
    for chunk in chunks:
        chunk_obj = wvc.data.DataObject(
            properties={
                "content": chunk['content'],
                "tokens": chunk['tokens'],
                "title": chunk['title'],
                "type": chunk['type'],
                "url": chunk['url'],
                "label": chunk['label']
            }
        )    
        chunk_objs.append(chunk_obj)
        
    with client:
        collection.data.insert_many(chunk_objs)
        
    logger.info(f"Loaded {len(chunks)} chunks into Weaviate")
    
def get_collection(client: WeaviateClient, collection_name: str | constants.WEAVIATE_COLLECTION_NAME)-> Collection:
    try:
        logger.info(f"Getting collection {collection_name}")
        collection = client.collections.get(collection_name)
    except Exception as e:
        logger.error(f"Collection {collection_name} not found")
        logger.info(f"Creating collection {collection_name}")
        collection = create_collection(client, collection_name)
    finally:
        return collection

def main(client: WeaviateClient, name: str) -> Collection:
    weaviateCollection = get_collection(client, name)
    
    elements = partition_pdf_elements_basic(constants.EMBEDDED_DOCS_DIR)
    
    elementDictionary = [element.to_dict() for element in elements]
    
    elementEmbeddings = []
    for element in elementDictionary:
        response = embeddingClient.embeddings(model=constants.DEFAULT_EMBEDDING_MODEL, prompt=element['text'])
        embedding = response["embedding"]
        elementEmbeddings.append(embedding)
        
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
        for element, embedding in zip(elementDictionary, elementEmbeddings)
    ]
    
    load_chunks_into_weaviate(chunk_embeddings_with_metadata, client, weaviateCollection.name)
    logger.info("Pipeline complete")
    return weaviateCollection