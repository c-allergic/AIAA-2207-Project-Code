import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 设置随机种子保证可重复性
np.random.seed(42)

# 创建数据集和特征信息的字典
datasets_info = {
    'Wine': {
        'data': datasets.load_wine(),
        'title': 'SVM Kernel Comparison on Wine Dataset (Alcohol vs Color)',
        'features': ['Feature 1', 'Feature 2'],
        'class_0': 0,
        'class_1': 1
    },
    'Breast Cancer': {
        'data': datasets.load_breast_cancer(),
        'title': 'SVM Kernel Comparison on Breast Cancer Dataset (Radius vs Area)',
        'features': ['Feature 1', 'Feature 2'],
        'class_0': 0,
        'class_1': 1
    },
    'Iris': {
        'data': datasets.load_iris(),
        'title': 'SVM Kernel Comparison on Iris Dataset (Sepal Features)',
        'features': ['Feature 1', 'Feature 2'],
        'class_0': 0,
        'class_1': 1
    }
}

# 定义核函数和参数
kernels = [
    ('Linear Kernel', {'kernel': 'linear'}),
    ('Polynomial Kernel (degree=3)', {'kernel': 'poly', 'degree': 3}),
    ('RBF Kernel', {'kernel': 'rbf'}),
    ('Sigmoid Kernel', {'kernel': 'sigmoid'})
]

# 处理每个数据集
for dataset_name, dataset_info in datasets_info.items():
    # 创建2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # 设置图表标题
    plt.suptitle(dataset_info['title'], fontsize=18, y=0.98)
    
    # 准备数据
    dataset = dataset_info['data']
    X, y = dataset.data, dataset.target
    if len(np.unique(y)) > 2:  # 简化为二分类问题
        mask = np.logical_or(y == dataset_info['class_0'], y == dataset_info['class_1'])
        X = X[mask]
        y = y[mask]
        y = (y == dataset_info['class_1']).astype(int)  # 转换为0和1
    
    # 拆分训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # 数据预处理
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2))
    ])
    X_train_proj = pipe.fit_transform(X_train)
    X_test_proj = pipe.transform(X_test)
    X_proj = np.vstack([X_train_proj, X_test_proj])
    y_combined = np.hstack([y_train, y_test])
    
    # 生成网格数据用于绘制决策边界
    x_min, x_max = X_proj[:, 0].min() - 1, X_proj[:, 0].max() + 1
    y_min, y_max = X_proj[:, 1].min() - 1, X_proj[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # 处理每个核函数
    for i, (kernel_name, params) in enumerate(kernels):
        ax = axes[i]
        
        # 训练SVM模型
        model = SVC(**params, gamma='scale', C=1.0)
        model.fit(X_train_proj, y_train)
        
        # 计算训练和测试准确率
        y_train_pred = model.predict(X_train_proj)
        y_test_pred = model.predict(X_test_proj)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        # 预测网格点
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界和散点图
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlGn)
        
        # 分别绘制训练集和测试集的点
        train_len = len(X_train_proj)
        markers = ['o', 's']  # 圆形和方形
        
        for cls in np.unique(y_combined):
            # 训练集点 (圆形)
            train_idx = (y_train == cls)
            ax.scatter(X_train_proj[train_idx, 0], X_train_proj[train_idx, 1], 
                      c='green' if cls == 0 else 'red', marker='o', 
                      edgecolors='k', alpha=0.8)
            
            # 测试集点 (方形)
            test_idx = (y_test == cls)
            ax.scatter(X_test_proj[test_idx, 0], X_test_proj[test_idx, 1], 
                     c='green' if cls == 0 else 'red', marker='s', 
                     edgecolors='k', alpha=0.8)
        
        # 添加标题和标签
        ax.set_title(f'{kernel_name}\nTrain Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}')
        ax.set_xlabel(dataset_info['features'][0])
        ax.set_ylabel(dataset_info['features'][1])
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()