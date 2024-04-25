# Description: Pipeline for the PHB-RAG project
import os
import holodeck.utilities.constants as constants
from loguru import logger
from typing import List
from unstructured.partition.pdf import partition_pdf

def partition_pdf_elements_basic(file_path) -> List:
    for filename in os.listdir(file_path):
        file = os.path.join(file_path, filename)
        if filename.endswith(".pdf"):
            logger.info(f"Partitioning file {file}...")
            return partition_pdf(file)

def partition_pdf_elements_complex(file_path) -> List:
    for filename in os.listdir(file_path):
        if filename.endswith(".pdf"):
            logger.info(f"Partitioning file {filename}...")
            return partition_pdf(filename, strategy="hi_res", infer_table_structure=True)
