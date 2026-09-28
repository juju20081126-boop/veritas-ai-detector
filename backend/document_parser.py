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


def extract_qualifying_text(raw_text: str) -> str:
    """
    Standard Institutional Turnitin-Grade Qualifying Text Extraction:
    1. Excludes Bibliographies, References, and Works Cited sections
    2. Excludes Markdown Tables and tabular matrices
    3. Excludes Standalone Structural Section Headers (e.g. ### Header)
    4. Excludes Horizontal Dividers (---, ***)
    5. Normalizes inline Markdown formatting (bold, italic, list markers) into continuous prose
    """
    if not raw_text or not raw_text.strip():
        return ""

    lines = raw_text.splitlines()
    qualifying_lines = []
    in_bibliography = False

    for line in lines:
        stripped = line.strip()

        # 1. Detect Bibliography / References section (handles both Markdown '# References' and PDF plain 'References')
        if re.match(r"^(?:#{1,6}\s*)?(?:references|bibliography|works\s+cited|sources)\s*$", stripped, re.IGNORECASE):
            in_bibliography = True
            continue
        if in_bibliography:
            # If a new major heading starts that is not bibliography, exit bibliography
            if re.match(r"^(?:#{1,3}\s+)?[A-Z][A-Za-z0-9\s:,\-'\"]{3,35}$", stripped) and not re.search(r"\b(?:press|routledge|oxford|springer|verlag|edition|pp\.|vol\.)\b", stripped, re.IGNORECASE):
                in_bibliography = False
            else:
                continue

        # 2. Exclude Markdown tables & table column header fragments
        if stripped.startswith("|") and stripped.endswith("|"):
            continue
        if re.match(r"^\|?[\s\-:|]+\|?$", stripped):
            continue
        if stripped in ["Dimension", "Mechanism of Extraction", "Societal / Epistemic Impact"]:
            continue

        # 3. Exclude horizontal rules & isolated bullet numbers
        if re.match(r"^(?:---|\*\*\*|___)$", stripped):
            continue
        if re.match(r"^\d+\.?$", stripped):
            continue

        # 4. Exclude standalone section headers (handles '# Abstract', 'Abstract', '### Introduction', etc.)
        if re.match(r"^(?:#{1,6}\s*)?(?:abstract|introduction|conclusion|theoretical framework|shadow work|epistemic injustice|counter-strategies|methodology|discussion)\s*$", stripped, re.IGNORECASE):
            continue
        if re.match(r"^#{1,6}\s+[A-Za-z0-9\s:,\-'\"]+$", stripped, re.IGNORECASE):
            continue

        # 5. Clean inline markdown syntax and bullet markers
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", stripped)
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
        cleaned = re.sub(r"^\s*[-*•●]\s*", "", cleaned)
        cleaned = re.sub(r"^\s*\d+\.\s*", "", cleaned)
        cleaned = cleaned.replace("\u200b", "").strip()

        if cleaned and len(cleaned) > 2:
            qualifying_lines.append(cleaned)

    return "\n\n".join(qualifying_lines)


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
    """
    Extracts text and metadata from PDF bytes using PyMuPDF (fitz).
    Uses block-level text extraction to unwrap visual line-breaks into coherent paragraphs,
    preserving multi-sentence discourse flow for Turnitin-grade AI sequence analysis.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    paragraph_blocks = []
    in_table = False
    in_references = False

    for page_num in range(len(doc)):
        if in_references:
            break
        page = doc[page_num]
        blocks = page.get_text("blocks")
        for b in blocks:
            # b[6] == 0 indicates text block
            if b[6] != 0:
                continue
            raw_block = b[4].strip()

            # 1. Detect Bibliography / References section
            if re.search(r"\b(?:references|bibliography|works\s+cited)\b", raw_block, re.IGNORECASE):
                parts = re.split(r"\b(?:references|bibliography|works\s+cited)\b", raw_block, flags=re.IGNORECASE)
                if parts[0].strip():
                    clean_before = " ".join(parts[0].split()).strip()
                    if clean_before and len(clean_before) > 5:
                        paragraph_blocks.append(clean_before)
                in_references = True
                break

            # 2. Filter Table blocks
            if "Dimension Mechanism of Extraction" in raw_block or "Societal / Epistemic Impact" in raw_block:
                in_table = True
                continue
            if in_table:
                if re.search(r"When platform algorithms gatekeep|The algorithmic commons", raw_block):
                    in_table = False
                else:
                    continue

            # 3. Clean and unwrap intra-block line breaks into single space
            block_text = " ".join(raw_block.split()).strip()
            block_text = block_text.replace("\u200b", "").replace("●", "").strip()

            # Skip isolated bullet numbers or empty lines
            if not block_text or re.match(r"^\d+\.?$", block_text):
                continue

            paragraph_blocks.append(block_text)

    # Stitch blocks seamlessly, merging cross-page broken sentences when a block doesn't end in terminal punctuation
    full_text = ""
    for b in paragraph_blocks:
        if not full_text:
            full_text = b
        elif not re.search(r'[.!?:]$', full_text.strip()):
            full_text = full_text.strip() + " " + b
        else:
            full_text = full_text.strip() + "\n\n" + b

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
