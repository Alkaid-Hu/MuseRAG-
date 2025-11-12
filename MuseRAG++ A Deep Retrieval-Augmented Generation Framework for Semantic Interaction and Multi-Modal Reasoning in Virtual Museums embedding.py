import os
import faiss
import torch
import numpy as np
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
from tqdm import tqdm
import json
import re

# Step 1: 初始化 BGE-M3 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

# Step 2: 向量化函数
def get_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :]  # CLS token
    return embeddings.squeeze().cpu().numpy()

# Step 3: 从PDF读取文本
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return full_text.strip()

# Step 4: 分段函数（简单分段策略）
def split_into_passages(text: str, min_len: int = 100, max_len: int = 300) -> List[str]:
    # 使用句号或换行拆分段落
    raw_passages = re.split(r"(?<=[。！？])\s*", text)
    passages, current = [], ""
    for sentence in raw_passages:
        if len(current) + len(sentence) < max_len:
            current += sentence
        else:
            if len(current) >= min_len:
                passages.append(current.strip())
            current = sentence
    if len(current) >= min_len:
        passages.append(current.strip())
    return passages

# Step 5: 处理PDF并获取向量和元数据
def vectorize_pdfs(pdf_folder: str) -> Tuple[List[np.ndarray], List[Tuple[str, str]]]:
    vectors = []
    metadata = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    for filename in tqdm(pdf_files, desc="Vectorizing Paragraphs", ncols=80):
        path = os.path.join(pdf_folder, filename)
        title = os.path.splitext(filename)[0]
        content = extract_text_from_pdf(path)
        passages = split_into_passages(content)

        for passage in passages:
            if len(passage.strip()) == 0:
                continue
            vec = get_embedding(passage)
            vectors.append(vec)
            metadata.append((title, passage))
    
    return vectors, metadata

# Step 6: 构建FAISS索引
def build_and_save_faiss_index(vectors: List[np.ndarray], metadata: List[Tuple[str, str]],
                                index_path: str, meta_path: str):
    dim = vectors[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.vstack(vectors))
    faiss.write_index(index, index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        for title, content in metadata:
            f.write(json.dumps({"title": title, "content": content}, ensure_ascii=False) + "\n")

    print(f"[✓] FAISS index saved to {index_path}")
    print(f"[✓] Metadata saved to {meta_path}")

# Step 7: 运行主程序
if __name__ == "__main__":
    pdf_dir = ""  # ← 修改为你的PDF文件夹路径
    index_output_path = "bge_m3_index.faiss"
    metadata_output_path = "bge_m3_metadata.jsonl"

    vecs, metas = vectorize_pdfs(pdf_dir)
    build_and_save_faiss_index(vecs, metas, index_output_path, metadata_output_path)
