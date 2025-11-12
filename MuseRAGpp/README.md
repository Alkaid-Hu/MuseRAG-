# MuseRAG++ (Minimal Reproducible Repo)
**A Deep Retrieval-Augmented Generation Framework for Semantic Interaction and Multi-Modal Reasoning in Virtual Museums**

> This repository provides a minimal-yet-professional implementation aligned with the paper *“MuseRAG++: A Deep Retrieval-Augmented Generation Framework for Semantic Interaction and Multi-Modal Reasoning in Virtual Museums”*.
> It focuses on **document indexing (BGE-M3, cosine/IP)**, **intent-agnostic dense retrieval**, and **provenance-aware generation** with inline citations `[S1]…`, plus **citation-coverage** metrics.

## ✨ Features
- **Robust indexing**: token sliding-window, metadata (doc_id/page/chunk/modality).
- **Stable retrieval**: unit-norm embeddings + **FAISS IndexFlatIP** (cosine).
- **Provenance-aware decoding**: inline citations `"[S1]"` with coverage metric.
- **Config-driven**: YAML configs for retrieval & generation.
- **Clean I/O**: JSONL metadata; one-command build and inference.

## 📦 Installation
```bash
git clone https://github.com/yourname/MuseRAGpp.git
cd MuseRAGpp
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🔐 API Keys
For Kimi (Moonshot) via OpenAI-compatible endpoint:
```bash
export DASHSCOPE_API_KEY=sk-xxxx
```

## 🧭 Quick Start
1) Put your PDFs into `examples/pdfs/`.
2) Build index:
```bash
python scripts/build_index.py
```
3) Run end-to-end inference:
```bash
python scripts/infer.py --query "长信宫灯如何减少烟尘？"
```

## ⚙️ Configs
- `configs/retrieval.yaml`: model name, chunking, index/meta outputs, topk.
- `configs/generation.yaml`: provider/model/temperature, min coverage (for reporting).

## 🧪 Metrics
- **Citation Coverage**: ratio of sentences that contain at least one `[S#]` citation.

## 🧩 Method ↔ Code Map
- *Hybrid Retrieval (dense baseline)* → `museragpp/embedding.py`, `search_generate.py`
- *Provenance-Aware Decoding* → `search_generate.py::make_prompt`, inline `[S#]` + coverage
- *Evaluation* → `museragpp/metrics.py`
> (Optional modules for **contrastive intent modeling**, **hybrid sparse+dense**, and **multimodal gating** can be added later; the repo already reserves the structure for easy extension.)

## 📄 Data Schema (metadata JSONL)
Each line:
```json
{"doc_id":"CXPL_0001","title":"CXPL_0001","page":3,"chunk_id":0,
 "content":"...","modality":"text","source":"xxx.pdf#p3"}
```

## 📜 License
MIT (or your preferred license).

## 🙌 Acknowledgements
BAAI for BGE models; FAISS; Transformers.