import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from feature_extract import data_dir
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# 读取数据并准备特征和标签
data = pd.read_csv(f'{data_dir}/dataset_processed.csv')
features = data.drop(['subjectkey', 'eventname', 'label'], axis=1)
labels = data['label']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=15)

# 过采样前的标签分布
print(f'Before over sampling: {Counter(y_train.values)}')

ros = RandomOverSampler(random_state=15)
X_train, y_train = ros.fit_resample(X_train, y_train)

# 过采样后的标签分布
print(f'After over sampling: {Counter(y_train.values)}')

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建MinMaxScaler对象，用于归一化特征
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train_scaled)
X_test_scaled = scaler.transform(X_test_scaled)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 转换为PyTorch的Tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32, device=device).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32, device=device).view(-1, 1)

# 构建自定义的神经网络模型
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 实例化神经网络模型
model = NeuralNet(X_train.shape[1])
model.to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模型训练
num_epochs = 15
batch_size = 256

for epoch in range(num_epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        targets = y_train_tensor[i:i+batch_size]

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 每个epoch打印训练损失
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 模型评估
with torch.no_grad():
    outputs = model(X_test_tensor)
    acc, f1, auc = accuracy_score(y_test_tensor.cpu().numpy(), outputs.cpu().numpy().round()), f1_score(y_test_tensor.cpu().numpy(), outputs.cpu().numpy().round()), roc_auc_score(y_test_tensor.cpu().numpy(), outputs.cpu().numpy().round())
    print(f'Accuracy: {acc}, F1: {f1}, AUC: {auc}')
