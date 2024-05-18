import os
import re
import holodeck.utilities.constants as constants
from loguru import logger
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Text, Element
from tqdm import tqdm

def partition_pdf_elements_basic(file_path) -> List:
    if file_path.endswith(".pdf"):
        logger.info(f"Partitioning file {file_path}...")
        return partition_pdf(file_path)

def partition_pdf_elements_complex(file_path) -> List:
    if file_path.endswith(".pdf"):
        logger.info(f"Partitioning file {file_path}...")
        return partition_pdf(file_path, strategy="hi_res", infer_table_structure=True)

def basic_overlap_chunking(partitioned_elements) -> List:
    chunks = chunk_elements(partitioned_elements,overlap=10,overlap_all=True)
    return chunks

def by_title_chunking(partitioned_elements) -> List:
    chunks = chunk_by_title(partitioned_elements,multipage_sections=False)
    return chunks

def remove_new_line_hyphens(chunks: List[Element]) -> List[Element]:
    remove_hyphens = lambda text: re.sub(r'(\w+)-\s*(\w+)', r'\1\2', text)
    for chunk in tqdm(chunks, desc="Applying regex sub", unit="chunk"):
        if isinstance(chunk, Text):
            chunk.apply(remove_hyphens)
    return chunks