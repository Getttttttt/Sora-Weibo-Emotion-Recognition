from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 加载数据
data_path = './ElasticNetLogistic/predict_label_data_normalize.csv'
data = pd.read_csv(data_path)

# 选择数值型变量作为特征
features = data.select_dtypes(include=[np.number]).drop(columns=['Unnamed: 0'])

# 对目标变量进行编码
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['emo_label'])

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 设置Elastic Net Logistic Regression的参数
model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.5, max_iter=10000, multi_class='ovr')

# 参数网格
param_grid = {
    'C': np.logspace(-4, 4, 20),
    'l1_ratio': np.linspace(0, 1, 10)
}

# 使用交叉验证和网格搜索优化参数
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 最优参数和模型评分
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best parameters:", best_params)
print("Best cross-validation score:", best_score)

# 使用最佳参数和整个训练集重新训练模型
model.set_params(**best_params)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 保存最优参数和评分
with open('model_performance.txt', 'w') as f:
    print("Best parameters:", best_params, file=f)
    print("Best cross-validation score:", best_score, file=f)
    print("\nClassification report:\n", classification_report(y_test, y_pred), file=f)

# 混淆矩阵
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
ax.matshow(conf_mat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_mat.shape[0]):
    for j in range(conf_mat.shape[1]):
        ax.text(x=j, y=i, s=conf_mat[i, j], va='center', ha='center')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
