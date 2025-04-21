# 电影评论情感分析工具

这是一个基于SVM的电影评论情感分析工具，可以对电影评论进行正面/负面情感分类。

## 功能特点

- 支持单条评论情感分析
- 支持批量评论情感分析
- 交互式命令行界面
- 提供预测置信度
- 支持英文评论

## 安装依赖

```bash
pip install scikit-learn pandas numpy nltk joblib matplotlib seaborn
```

## 使用方法

### 1. 交互式模式（默认）

```bash
python movie_sentiment_cli.py
```

### 2. 单条评论分析

```bash
python movie_sentiment_cli.py --mode single
```

### 3. 批量分析

从文件读取评论：
```bash
python movie_sentiment_cli.py --mode batch --input reviews.txt
```

手动输入多条评论：
```bash
python movie_sentiment_cli.py --mode batch
```

### 4. 使用自定义模型

```bash
python movie_sentiment_cli.py --model path/to/your/model.joblib
```

## 文件说明

- `SVM.py`: 模型训练代码
- `predict.py`: 预测功能实现
- `movie_sentiment_cli.py`: 命令行界面
- `sentiment_svm_model.joblib`: 预训练模型


## 注意事项

1. 确保已安装所有必要的依赖包
2. 首次运行时会自动下载NLTK数据
3. 支持英文评论
4. 建议评论长度在10-500字之间 