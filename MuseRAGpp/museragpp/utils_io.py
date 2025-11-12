import re, json
from typing import List, Tuple, Dict
from pypdf import PdfReader

def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def read_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    pages = []
    r = PdfReader(pdf_path)
    for i, p in enumerate(r.pages, start=1):
        txt = clean_text(p.extract_text())
        if txt:
            pages.append((i, txt))
    return pages

def write_jsonl(path: str, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]