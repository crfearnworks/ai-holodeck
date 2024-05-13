import os
import holodeck.utilities.constants as constants
from loguru import logger
from typing import List
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title

def basic_overlap_chunking(partitioned_elements) -> List:
    chunks = chunk_elements(partitioned_elements,overlap=10,overlap_all=True)
    return chunks

def by_title_chunking(partitioned_elements) -> List:
    chunks = chunk_by_title(partitioned_elements,multipage_sections=False)
    return chunks

