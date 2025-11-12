import os, faiss, yaml, numpy as np, torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from .utils_io import read_jsonl
from .metrics import citation_coverage

Q_PROMPT = "Represent this question for retrieving relevant passages:"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_pool(h, mask):
    m = mask.unsqueeze(-1).expand(h.size()).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)

@torch.no_grad()
def encode_query(model, tok, q: str) -> np.ndarray:
    batch = tok([f"{Q_PROMPT} {q}"], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    h = model(**batch).last_hidden_state
    z = mean_pool(h, batch["attention_mask"])
    z = torch.nn.functional.normalize(z, p=2, dim=1)
    return z.cpu().numpy().astype(np.float32)

def search(q: str, cfg_retrieval="configs/retrieval.yaml", topk=None) -> List[Dict]:
    cfg = yaml.safe_load(open(cfg_retrieval))
    topk = topk or cfg["topk"]
    model_name = cfg["model_name"]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    qv = encode_query(model, tok, q)

    index = faiss.read_index(cfg["index_out"])
    meta = read_jsonl(cfg["meta_out"])
    scores, idxs = index.search(qv, topk)
    hits = []
    for r, (i, s) in enumerate(zip(idxs[0], scores[0]), start=1):
        m = meta[i]
        hits.append({
            "rank": r, "score": float(s), "doc_id": m["doc_id"],
            "title": m["title"], "page": m["page"], "chunk_id": m["chunk_id"],
            "content": m["content"], "source": m["source"]
        })
    return hits

def build_snippets(hits: List[Dict]) -> str:
    return "\n".join([f"[S{i}] ({h['source']}) {h['content']}" for i, h in enumerate(hits, 1)])

def make_prompt(query: str, snippets: str) -> str:
    return (
        "You are a provenance-aware assistant. Answer using ONLY the numbered snippets. "
        "Cite snippet IDs like [S1],[S2] after each claim.\n\n"
        f"Question: {query}\n\nSnippets:\n{snippets}\n\nAnswer:"
    )

def generate(query: str, hits: List[Dict], cfg_generation="configs/generation.yaml") -> Dict:
    cfg = yaml.safe_load(open(cfg_generation))
    provider = cfg.get("provider", "kimi")
    model = cfg.get("model")
    temp  = float(cfg.get("temperature", 0.2))

    if provider == "kimi":
        client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    else:
        client = OpenAI()

    prompt = make_prompt(query, build_snippets(hits))
    rsp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content": prompt}],
        temperature=temp
    )
    answer = rsp.choices[0].message.content.strip()
    cov = citation_coverage(answer)
    sources = [f"[S{i}] {h['title']} (p.{h['page']}) — {h['source']}" for i, h in enumerate(hits, 1)]
    return {"answer": answer, "coverage": cov, "sources": list(dict.fromkeys(sources))}