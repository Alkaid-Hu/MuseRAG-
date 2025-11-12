import os, numpy as np, torch, faiss, yaml
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
from .utils_io import read_pdf_pages, write_jsonl

DOC_PROMPT = "Represent this document passage for retrieval:"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_pool(h, mask):
    m = mask.unsqueeze(-1).expand(h.size()).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)

@torch.no_grad()
def encode(model, tok, texts: List[str], max_len=512) -> np.ndarray:
    batch = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    h = model(**batch).last_hidden_state
    emb = mean_pool(h, batch["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype(np.float32)

def split_by_tokens(tok, text: str, max_tokens=220, stride=60) -> List[str]:
    toks = tok.tokenize(text)
    chunks = []
    step = max_tokens - stride
    for s in range(0, len(toks), max(1, step)):
        sub = toks[s:s+max_tokens]
        if not sub: break
        chunks.append(tok.convert_tokens_to_string(sub).strip())
        if s + max_tokens >= len(toks): break
    return [c for c in chunks if c]

def build_index(cfg_path="configs/retrieval.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    model_name = cfg["model_name"]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    vecs, metas = [], []
    for fn in os.listdir(cfg["pdf_dir"]):
        if not fn.lower().endswith(".pdf"): continue
        fpath = os.path.join(cfg["pdf_dir"], fn)
        doc_id = os.path.splitext(fn)[0]
        for page_no, text in read_pdf_pages(fpath):
            for j, chunk in enumerate(split_by_tokens(tok, text, cfg["chunk_tokens"], cfg["chunk_stride"])):
                vec = encode(model, tok, [f"{DOC_PROMPT} {chunk}"], cfg["max_length"])[0]
                vecs.append(vec)
                metas.append({
                    "doc_id": doc_id,
                    "title": doc_id,
                    "page": page_no,
                    "chunk_id": j,
                    "content": chunk,
                    "modality": "text",
                    "source": f"{fn}#p{page_no}"
                })

    V = np.vstack(vecs).astype(np.float32)
    index = faiss.IndexFlatIP(V.shape[1])  # Cosine via unit norm
    index.add(V)
    faiss.write_index(index, cfg["index_out"])
    write_jsonl(cfg["meta_out"], metas)
    print(f"[✓] index @ {cfg['index_out']}  |  meta @ {cfg['meta_out']}  |  vecs={len(V)}")