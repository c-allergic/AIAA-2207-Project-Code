import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from predict import preprocess, clean_text

def load_model(model_path):
    """加载训练好的模型"""
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

def predict_sentiment(model, text):
    """预测单个评论的情感"""
    processed_text = preprocess(text)
    prob = model.decision_function([processed_text])[0]
    pred = "正面" if model.predict([processed_text])[0] == 1 else "负面"
    confidence = abs(prob)
    return pred, confidence

def batch_predict(model, texts):
    """批量预测多个评论的情感"""
    results = []
    for text in texts:
        pred, conf = predict_sentiment(model, text)
        results.append({
            '评论': text,
            '预测': pred,
            '置信度': conf
        })
    return pd.DataFrame(results)

def plot_confusion_matrix(y_true, y_pred):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='电影评论情感分析工具')
    parser.add_argument('--model', default='sentiment_svm_model.joblib',
                      help='模型文件路径')
    parser.add_argument('--mode', choices=['single', 'batch', 'interactive'],
                      default='interactive', help='运行模式')
    parser.add_argument('--input', help='输入文件路径（批量模式）')
    
    args = parser.parse_args()
    
    # 加载模型
    model = load_model(args.model)
    if model is None:
        return
    
    if args.mode == 'single':
        # 单条评论预测模式
        review = input("请输入电影评论: ")
        pred, conf = predict_sentiment(model, review)
        print(f"\n预测结果: {pred}")
        print(f"置信度: {conf:.2f}")
        
    elif args.mode == 'batch':
        # 批量预测模式
        if args.input:
            try:
                with open(args.input, 'r', encoding='utf-8') as f:
                    reviews = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"读取文件时出错: {e}")
                return
        else:
            print("请输入评论（每行一条，输入空行结束）：")
            reviews = []
            while True:
                line = input()
                if not line:
                    break
                reviews.append(line)
        
        results = batch_predict(model, reviews)
        print("\n预测结果：")
        print(results.to_string(index=False))
        
    else:  # interactive mode
        print("欢迎使用电影评论情感分析工具！")
        print("输入 'q' 退出")
        while True:
            review = input("\n请输入电影评论: ")
            if review.lower() == 'q':
                break
            pred, conf = predict_sentiment(model, review)
            print(f"预测结果: {pred}")
            print(f"置信度: {conf:.2f}")

if __name__ == '__main__':
    main() 