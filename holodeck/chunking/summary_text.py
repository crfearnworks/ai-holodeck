import os
from pypdf import PdfReader 
import nltk
from tqdm import tqdm
import numpy as np
import spacy
from typing import List
from loguru import logger
from holodeck.ollama import ollama_utils

def extract_text_from_pdf(file_path) -> str:
    logger.info(f"Extracting text from {file_path}...")
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        text = " ".join(page.extract_text() for page in pdf.pages)
    logger.info(f"Extracted text from {len(pdf.pages)} pages from {file_path}.")
    return text

def spacy_process(nlp, text):
    doc = nlp(text)
    sents = list(doc.sents)
    vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])

    return sents, vecs

def spacy_cluster_text(sents, vecs, threshold) -> List[List[int]]:
    clusters = [[0]]
    for i in range(1, len(sents)):
        if np.dot(vecs[i], vecs[i-1]) < threshold:
            clusters.append([])
        clusters[-1].append(i)
    
    return clusters

def spacy_clean_text(text):
    # Add your text cleaning process here
    return text

def spacy_chunk_text(text) -> List:
    logger.info("Loading spacy natural language model and allow for maximum string length") 
    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 30000000

    logger.info("Initialize the clusters lengths list and final texts list")
    clusters_lens = []
    final_texts = []

    logger.info("Process the large chunk of text")
    threshold = 0.3
    sents, vecs = spacy_process(nlp, text)

    logger.info("Cluster the sentences")
    clusters = spacy_cluster_text(sents, vecs, threshold)

    for cluster in clusters:
        cluster_txt = spacy_clean_text(' '.join([sents[i].text for i in cluster]))
        cluster_len = len(cluster_txt)
        
        logger.info("Check if the cluster is too short or too long")
        if cluster_len < 60:
            continue
        
        elif cluster_len > 3000:
            threshold = 0.6
            sents_div, vecs_div = spacy_process(nlp, cluster_txt)
            reclusters = spacy_cluster_text(sents_div, vecs_div, threshold)
            
            for subcluster in reclusters:
                div_txt = spacy_clean_text(' '.join([sents_div[i].text for i in subcluster]))
                div_len = len(div_txt)
                
                if div_len < 60 or div_len > 3000:
                    continue
                
                clusters_lens.append(div_len)
                final_texts.append(div_txt)
                
        else:
            clusters_lens.append(cluster_len)
            final_texts.append(cluster_txt)
    
    return final_texts

def spacy_chunk_and_summarize(chunks, o_client: ollama_utils.OllamaClient) -> List:
    logger.info("Summarizing chunks")
    task = "Summarize this in one to three sentences."
    summaries_and_chunks = []
    for chunk in tqdm(enumerate(chunks)):
        summary = o_client.generate(
            model='mistral:7b',
            prompt=f"Using the following context: {chunk}, perform this task: {task}"
        )
        summary_response = summary['response']
        summaries_and_chunks.append({'summary': summary_response, 'chunk': chunk})
    return summaries_and_chunks

def summary_text_pipeline(file_name, o_client) -> List:
    text = extract_text_from_pdf(file_name)
    chunked_texts = spacy_chunk_text(text)
    summary_and_chunks = spacy_chunk_and_summarize(chunked_texts, o_client)
    return summary_and_chunks
    