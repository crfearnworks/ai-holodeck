import os
from llmsherpa.readers import LayoutPDFReader
from llmsherpa.readers.layout_reader import Document
from loguru import logger

def read_pdf(file_path: str, llmsherpa_url: str) -> Document:
    logger.info(f"Extracting text from {file_path}...")
    pdf_reader = LayoutPDFReader(parser_api_url=llmsherpa_url)
    doc = pdf_reader.read_pdf(file_path)
    return doc

def apply_OCR(llmsherpa_url: str) -> str:
    logger.info("Adding OCR capability...")
    logger.info(f"old API URL: {llmsherpa_url}")
    llmsherpa_url = f"{llmsherpa_url}&applyOcr=yes"
    logger.info(f"new API URL: {llmsherpa_url}")
    return llmsherpa_url

def apply_new_indent_parser(llmsherpa_url: str) -> str:
    logger.info("Adding new indent parser...")
    logger.info(f"old API URL: {llmsherpa_url}")
    llmsherpa_url = f"{llmsherpa_url}&useNewIndentParser=yes"
    logger.info(f"new API URL: {llmsherpa_url}")
    return llmsherpa_url

