import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List
import json
import os
from openai import OpenAI

client = OpenAI(
    api_key="",  # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 加载 BGE-M3 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

# 编码函数
def get_embedding(text: str) -> np.ndarray:
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        model_output = model(**inputs)
        embeddings = model_output.last_hidden_state[:, 0, :]  # CLS token
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # L2 normalize
    return embeddings.cpu().numpy()[0]

# 查询函数
def search_faiss_index(query: str, top_k: int = 5):
    # 加载索引和元数据
    index = faiss.read_index("bge_m3_index.faiss")

    with open("bge_m3_metadata.jsonl", "r", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]
    
    # 查询向量化
    query_vec = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)
    
    # 返回匹配标题和距离
    results = [
    (metadata[i]['title'], metadata[i]['content'], distances[0][j])
    for j, i in enumerate(indices[0])
]

    return results
    

# 示例查询
query = "介绍长信宫灯。"
results = search_faiss_index(query, top_k=5)
if not results:
    print("未找到答案。")
else:
    # 拼接资料内容作为上下文
    context_texts = []
    sources = []
    for i, (title, content, dist) in enumerate(results):
        context_texts.append(f"资料{i+1}：{content}")
        # sources.append(f"{i+1}. {title}")

        sources.append(f"{title}")
    
    context_prompt = "\n".join(context_texts)
    full_prompt = f"根据以下资料，回答用户的问题。\n\n问题：{query}\n\n{context_prompt}\n\n如果资料中没有明确信息，请说明未找到答案。"

    completion = client.chat.completions.create(
        model="Moonshot-Kimi-K2-Instruct",
        messages=[{'role': 'user', 'content': full_prompt}]
    )

    print("【回答】")
    print(completion.choices[0].message.content.strip())

    print("\n【参考资料】")
    source_new=list(set(sources))
    print("\n".join(source_new))
