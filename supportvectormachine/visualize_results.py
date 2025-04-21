import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

# 加载模型
model_path = "sentiment_svm_model.joblib"
model = joblib.load(model_path)

# 加载测试数据
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# 确保NLTK数据已下载
nltk.download(['stopwords', 'wordnet', 'omw-1.4'])

# 文本预处理函数
def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r"[^a-zA-Z.!?']", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    cleaned = clean_text(text)
    words = cleaned.split()
    processed = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(processed)

# 加载数据集
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
df = pd.read_csv(path + '/IMDB Dataset.csv')

# 预处理数据
print("预处理数据...")
df['processed_review'] = df['review'].apply(preprocess)

# 特征工程
tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1,2),
    min_df=5,
    max_df=0.7
)

# 划分训练测试集
X = df['processed_review']
y = df['sentiment'].map({'positive':1, 'negative':0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用模型进行预测
y_pred = model.predict(X_test)
y_score = model.decision_function(X_test)

# 创建输出目录
os.makedirs("results", exist_ok=True)

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Movie Sentiment Classification Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 从混淆矩阵计算性能指标
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# 绘制模型性能图
plt.figure(figsize=(10, 8))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
plt.bar(metrics, values, color='skyblue')
plt.ylim(0, 1)
plt.title('SVM Model Performance Metrics')
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
plt.savefig('results/performance_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

print("可视化结果已保存到 results 目录") 