import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 假设您已经有了数据
# X 是特征数据，y 是标签
X = np.random.rand(100, 4)  # 示例：100个样本，每个样本4个特征
y = np.random.randint(0, 2, 100)  # 示例：二分类问题的标签

# 60-40分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.4,  # 40%作为测试集
    random_state=42  # 设置随机种子，确保结果可重复
)

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(
    n_estimators=100,  # 树的数量
    random_state=42
)

# 训练模型
rf_classifier.fit(X_train, y_train)

# 预测
y_pred = rf_classifier.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))
