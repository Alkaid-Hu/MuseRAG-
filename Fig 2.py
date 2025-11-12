import matplotlib.pyplot as plt

# 数据
models = [
    "BM25 + ChatGLM4-9B-Chat",
    "BGE m3 + ChatGLM4-9B-Chat",
    "ChatGLM4-9B-Chat",
    "Qwen3-8B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "MuseRAG++ (Ours)"
]

recall = [74.2, 77.8, 75.3, 71.3, 73.5, 83.1]
precision = [69.1, 71.0, 70.5, 68.9, 70.2, 76.4]
accuracy = [66.5, 69.4, 68.7, 65.7, 68.0, 74.5]
f1 = [71.6, 74.2, 72.8, 70.1, 71.8, 79.6]

# 颜色和线型
plt.figure(figsize=(8, 5))
plt.plot(models, recall, marker='o', label='Recall@10')
plt.plot(models, precision, marker='s', label='Precision')
plt.plot(models, accuracy, marker='^', label='Accuracy')
plt.plot(models, f1, marker='D', label='F1-score')

# 样式设置
plt.xlabel("Models", fontsize=12)
plt.ylabel("Scores (%)", fontsize=12)
plt.title("Comparative Performance across Models", fontsize=14, weight='bold')
plt.xticks(rotation=25, ha='right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()

# 保存图像
plt.savefig("comparative_results.png", dpi=400)
plt.show()
