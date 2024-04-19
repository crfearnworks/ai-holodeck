from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, NarrativeText, Text
from unstructured.chunking.basic import chunk_elements
import weaviate
import os
from weaviate.util import generate_uuid5
from typing import List
import ollama

#FILE_PATH = "docs/Player_s Handbook.pdf"
FILE_PATH = "docs/NIST.SP.800-171r2.pdf"

def process_pdf(file_path: str):
    # partition the pdf
    elements = partition_pdf(filename=file_path, strategy="fast")
    # convert elements into strings
    texts = [str(el) for el in elements]
    return texts

client = weaviate.connect_to_local()
collection = client.collections.create(name="PHB")

docs = process_pdf(FILE_PATH)

for i, d in enumerate(docs):
    response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
    embedding = response["embedding"]
    collection.batch(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[d]
    )
    
# an example prompt
prompt = "Explain the fireball spell."

# generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
    prompt=prompt,
    model="mxbai-embed-large"
)
results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=1
)
data = results['documents'][0][0]

# generate a response combining the prompt and data we retrieved in step 2
output = ollama.generate(
    model="mistral",
    prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

print(output['response'])

client.close()