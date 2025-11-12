import matplotlib.pyplot as plt
import numpy as np

# 模型名称
models = [
    "w/o User Intent\nModeling",
    "w/o Hybrid\nRetrieval",
    "w/o Multimodal\nGating",
    "w/o Provenance-\naware Decoding",
    "MuseRAG++\n(Full Model)"
]

# 指标数据
recall = [77.2, 78.6, 80.1, 82.0, 83.1]
precision = [71.8, 72.0, 73.5, 74.8, 76.4]
accuracy = [69.3, 70.2, 71.9, 73.0, 74.5]
f1 = [74.4, 75.2, 76.7, 78.2, 79.6]

# 柱宽与位置
x = np.arange(len(models))
width = 0.2

# 绘制柱状图
plt.figure(figsize=(9, 5))
plt.bar(x - 1.5*width, recall, width, label='Recall@10')
plt.bar(x - 0.5*width, precision, width, label='Precision')
plt.bar(x + 0.5*width, accuracy, width, label='Accuracy')
plt.bar(x + 1.5*width, f1, width, label='F1-score')

# 样式调整
plt.xlabel("Model Variants", fontsize=12)
plt.ylabel("Scores (%)", fontsize=12)
plt.title("Ablation Study of MuseRAG++", fontsize=14, weight='bold')
plt.xticks(x, models, rotation=20, ha='right')
plt.ylim(65, 85)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(fontsize=10, loc='upper left', frameon=False)
plt.tight_layout()

# 保存图像
plt.savefig("ablation_study_muserag.png", dpi=400)
plt.show()
