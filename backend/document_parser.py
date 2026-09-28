"""
Document Ingestion and Parsing Module
Supports multi-format parsing (.pdf, .docx, .txt, .md) and robust sentence boundary segmentation.
"""

import io
import re
from typing import List, Dict, Any, Tuple


# Common English honorifics and abbreviations to avoid false sentence boundaries
ABBREVIATIONS = (
    r"(?:Mr|Mrs|Ms|Dr|Prof|Rev|Capt|Lt|Col|Gen|Sgt|Rep|Sen|Gov|"
    r"vs|etc|e\.g|i\.e|et al|fig|dept|univ|jr|sr|inc|ltd|co|corp|"
    r"u\.s|u\.k|u\.n|e\.u|a\.m|p\.m|no|vol|pp|approx)"
)


def split_sentences(text: str) -> List[str]:
    """
    Splits text into coherent sentences without incorrectly breaking
    on common abbreviations, decimals, or ellipses.
    """
    if not text or not text.strip():
        return []
        
    text = text.strip()
    
    # 1. Mask periods in abbreviations temporarily
    def mask_abbrev(match):
        return match.group(0).replace(".", "§DOT§")
        
    # Mask decimals (e.g. 3.14)
    masked = re.sub(r'(\d+)\.(\d+)', r'\1§DOT§\2', text)
    
    # Mask abbreviations (e.g. Dr., e.g., etc.)
    masked = re.sub(rf'\b{ABBREVIATIONS}\.', mask_abbrev, masked, flags=re.IGNORECASE)
    
    # Mask ellipses (e.g. ...)
    masked = re.sub(r'\.{3,}', '§ELLIP§', masked)
    
    # 2. Split on sentence terminals followed by whitespace/newlines or end of text
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'“‘])|\n{2,}', masked)
    
    cleaned_sentences = []
    for s in raw_sentences:
        # Restore masked characters
        restored = s.replace("§DOT§", ".").replace("§ELLIP§", "...")
        restored = re.sub(r'\s+', ' ', restored).strip()
        if restored and len(restored) > 2:
            cleaned_sentences.append(restored)
            
    # Fallback if text is a single paragraph without terminal punctuation
    if not cleaned_sentences and text:
        cleaned_sentences = [re.sub(r'\s+', ' ', text).strip()]
        
    return cleaned_sentences


def parse_pdf(file_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """Extracts text and metadata from PDF bytes using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF
    
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_text = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        pages_text.append(page.get_text("text"))
        
    full_text = "\n\n".join(pages_text)
    metadata = {
        "page_count": len(doc),
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "format": "PDF"
    }
    return full_text, metadata


def parse_docx(file_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """Extracts text and metadata from DOCX bytes using python-docx."""
    import docx
    
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)
    
    metadata = {
        "paragraph_count": len(paragraphs),
        "format": "DOCX"
    }
    return full_text, metadata


def parse_uploaded_file(filename: str, file_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    Parses an uploaded file based on its extension.
    Returns (extracted_text, metadata).
    """
    fn = filename.lower()
    if fn.endswith(".pdf"):
        return parse_pdf(file_bytes)
    elif fn.endswith(".docx"):
        return parse_docx(file_bytes)
    elif fn.endswith(".txt") or fn.endswith(".md"):
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1", errors="replace")
        return text, {"format": "Plain Text"}
    else:
        # Fallback to UTF-8 decoding
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1", errors="replace")
        return text, {"format": "Unknown / Text"}
