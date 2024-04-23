# Description: Pipeline for the PHB-RAG project
import holodeck.utilities.constants as constants
from loguru import logger
from typing import List
from unstructured.partition.pdf import partition_pdf

def partition_pdf_elements_basic(file_path) -> List[Element]:
    logger.info(f"Partitioning file {file_path}...")
    return partition_pdf(file_path)

def partition_pdf_elements_complex(file_path) -> List[Element]:
    logger.info(f"Partitioning file {file_path}...")
    return partition_pdf(file_path, strategy="hi_res", infer_table_structure=True)
