import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

# 5. 用不同参数测试，找最优组合
print("=" * 50)
print("随机森林 不同参数的准确率")
print("=" * 50)

best_acc = 0
best_params = {}

for n_estimators in [50, 100, 200, 300]:
    for max_depth in [3, 5, 7, 10, None]:
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(x_train, y_train)
        acc = accuracy_score(y_test, rf.predict(x_test))
        depth_str = str(max_depth) if max_depth else "None"
        print(f"n_estimators={n_estimators:3d}, max_depth={depth_str:4s}  Accuracy: {acc*100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_params = {'n_estimators': n_estimators, 'max_depth': max_depth}

# 6. 用最优参数输出详细报告
print("=" * 50)
depth_str = str(best_params['max_depth']) if best_params['max_depth'] else "None"
print(f"最优参数: n_estimators={best_params['n_estimators']}, max_depth={depth_str}")
print(f"Accuracy: {best_acc*100:.2f}%")
print("=" * 50)

rf_best = RandomForestClassifier(**best_params, random_state=42)
rf_best.fit(x_train, y_train)
y_pred = rf_best.predict(x_test)
print(classification_report(y_test, y_pred))

# 7. 特征重要性
print("=" * 50)
print("特征重要性排名")
print("=" * 50)
feature_names = data.columns[:8]
importances = rf_best.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"  {name:20s} {imp:.4f}")
