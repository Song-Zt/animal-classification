import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 加载数据
data = pd.read_csv('/data/Diabetes.csv')
x, y = data.iloc[:, :8], data.iloc[:, 8]

# 2. 处理异常值（0替换为中位数）
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
x[zero_cols] = x[zero_cols].replace(0, np.nan)
x = x.fillna(x.median())

# 3. 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 4. 特征标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 5. 用不同K值测试，找最优K
print("=" * 40)
print("KNN 不同K值的准确率")
print("=" * 40)
best_acc = 0
best_k = 1
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    acc = accuracy_score(y_test, knn.predict(x_test))
    print(f"K={k:2d}  Accuracy: {acc*100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        best_k = k

# 6. 用最优K值输出详细报告
print("=" * 40)
print(f"最优 K={best_k}, Accuracy: {best_acc*100:.2f}%")
print("=" * 40)

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(x_train, y_train)
y_pred = knn_best.predict(x_test)
print(classification_report(y_test, y_pred))
