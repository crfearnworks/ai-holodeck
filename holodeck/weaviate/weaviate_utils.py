from itertools import count
import os
import weaviate
import weaviate.classes as wvc
from weaviate.collections import Collection
from weaviate.client import WeaviateClient
from holodeck.chunking.unstructured import basic_overlap_chunking, by_title_chunking, partition_pdf_elements_basic, remove_new_line_hyphens
from holodeck.chunking import summary_text
from holodeck.ollama.ollama_client import OllamaClient
import holodeck.utilities.constants as constants 
from typing import List, Dict
from loguru import logger

def create_weaviate_local_client() -> WeaviateClient:
    client = weaviate.connect_to_local(
        additional_config=wvc.init.AdditionalConfig(
            timeout=(60,1800),
        ),
    )
    return client

def create_collection(client: WeaviateClient, collection_name: str)-> Collection:
    with client: 
        logger.info(f"Creating collection {collection_name}")

        client.collections.create(
            name=collection_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(),
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE # select prefered distance metric
        ),
    )
    collection = client.collections.get(name=collection_name)
    return collection

def delete_collection(client: WeaviateClient, collection_name: str) -> None:
    with client:
        logger.info(f"Deleting collection {collection_name}")
        
        client.collections.delete(name=collection_name)

def load_chunks_into_weaviate(chunks: List, client: WeaviateClient, collection: Collection):

    logger.info("Chunking data into Weaviate")
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
    
def get_collection(client: WeaviateClient, collection_name = None)-> Collection:
    if collection_name is None:
        collection_name = constants.WEAVIATE_COLLECTION_NAME
    try:
        logger.info(f"Getting collection {collection_name}")
        collection = client.collections.get(collection_name)
    except Exception as e:
        logger.error(f"Collection {collection_name} not found")
        logger.info(f"Creating collection {collection_name}")
        collection = create_collection(client, collection_name)
    finally:
        return collection

def check_embedded_existance(client: WeaviateClient, collection: Collection, file_path: str, o_client: OllamaClient) -> List:
    with client:
        elements = []
        for filename in os.listdir(file_path):
            element = []
            chunks = []
            file = os.path.join(file_path, filename)
            logger.info(f"Checking if {filename} exists in Weaviate")
            try:
                objectsResults = collection.query.fetch_objects(return_properties=["title"])
                logger.info(f"Objects in Weaviate: {objectsResults}")
                objects = [] 
                for obj in objectsResults.objects:
                    objects.append(obj.properties['title'])
                logger.info(f"Objects in Weaviate: {objects}")
                directory = list(dict.fromkeys(objects))
                logger.info(f"Directory: {directory}")
                if filename not in directory:
                    logger.info(f"{filename} does not exist in Weaviate")
                    element = partition_pdf_elements_basic(file)
                    elements.extend(element)
                else:    
                    logger.info(f"{filename} exists in Weaviate")
            except weaviate.exceptions.WeaviateQueryError as e:
                logger.error(f"Check failed: {e}")
                element = partition_pdf_elements_basic(file)
                elements.extend(element)
                
        return elements

def generate_results_content(client: WeaviateClient, collection: Collection, query: str, resultsVectors: List) -> List:
    with client:
        logger.info(f"Querying Weaviate collection with query: {query}")
        resultsContent = []
        resultsReferences = []
        results = collection.query.near_vector(
            near_vector=resultsVectors,
        )
        for obj in results.objects:
            resultsContent.append(obj.properties['content'])
            resultsReferences.append(obj.properties['title'])
        resultsReferences = list(dict.fromkeys(resultsReferences))
        return resultsContent, resultsReferences