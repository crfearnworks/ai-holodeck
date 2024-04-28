# Description: Pipeline for the PHB-RAG project
import os
import holodeck.utilities.constants as constants
from loguru import logger
from typing import List
from unstructured.partition.pdf import partition_pdf

def partition_pdf_elements_basic(file_path) -> List:
    if file_path.endswith(".pdf"):
        logger.info(f"Partitioning file {file_path}...")
        return partition_pdf(file_path)

def partition_pdf_elements_complex(file_path) -> List:
    if file_path.endswith(".pdf"):
        logger.info(f"Partitioning file {file_path}...")
        return partition_pdf(file_path, strategy="hi_res", infer_table_structure=True)
