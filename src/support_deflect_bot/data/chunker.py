from typing import Dict, List


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Naive character-based chunker with overlap.
    Simple and effective for a weekend build.
    """
    text = text.strip()
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return [c for c in (c.strip() for c in chunks) if c]


def build_docs_from_files(files: Dict[str, str]) -> Dict[str, str]:
    """
    Given a mapping {path: text}, returns same (no transformation).
    Placeholder for future preprocessing (markdown cleanup, heading capture).
    """
    return files
