import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn, optim
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time

look_back = 12
look_forward = 6

k = 3

dims = [6, 12, 24]

#%%
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(look_back * hidden_dim, look_forward * output_dim)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output.reshape(output.size(0), -1)
        output = self.linear(output)
        output = output.reshape(output.size(0), look_forward, output_dim)
        return output

class MAML:
    def __init__(self, model, lr_inner=0.01, lr_outer=0.001, n_inner_loop=1):
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.n_inner_loop = n_inner_loop
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_outer)

    def inner_loop(self, data, targets):
        for step in range(self.n_inner_loop):
            preds = self.model(data)
            loss = nn.MSELoss()(preds, targets)
            self.model.zero_grad()
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            fast_weights = OrderedDict((name, param - self.lr_inner * grad)
                                       for ((name, param), grad) in zip(self.model.named_parameters(), grads))
            for name, param in self.model.named_parameters():
                param.data = fast_weights[name]
        return fast_weights

    def outer_loop(self, data, targets):
        fast_weights = self.inner_loop(data, targets)
        preds = self.model(data)
        loss = nn.MSELoss()(preds, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, sequence_length):
        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        return (self.data[index:index+self.sequence_length], self.targets[index+self.sequence_length])

#%%
# 选择device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
print('Using {} device'.format(device))

#%% 构造数据
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.cluster import KMeans
import copy
import re
import time

import sys
sys.path.append('STC-ML/')

#%% 读取dict数据格式
def load_data(filename):
    import pickle
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # 确定data['flow']长度为288的id列表
    id_list = [d['id'] for d in data]

    # 将data中的'flow'抽取为新的numpy数组
    data_flow = np.array([d['flow'] for d in data])

    return data_flow, id_list

#%% 加载数据并准备训练
data_flow, id_list = load_data('STC-ML/data/traffic_flow/1/20230306/part-00000.pkl')

#%% 读取feature
feature = pd.read_csv('STC-ML/data/link_feature.csv')

#%% 确定节点数和时间步数量
# 对data['flow']进行归一化
max_value_flow = np.max(data_flow)
data_flow = data_flow / max_value_flow

# 删除id_list不在feature['link_ID']中的条目，并删除data_flow对应的row
data_flow = data_flow[[i for i,x in enumerate(id_list) if x in feature['link_ID'].values]]
id_list = [i for i in id_list if i in feature['link_ID'].values]

time_len = data_flow.shape[1]
num_nodes = data_flow.shape[0]

#%%

# plit data_flow into training and testing sets
train_ratio = 0.8
train_size = int(time_len * train_ratio)
train_data = data_flow[:, :train_size]
test_data = data_flow[:, train_size:]
# Transpose train_data and test_data
train_data = np.transpose(train_data, (1, 0))
test_data = np.transpose(test_data, (1, 0))

#%%
# Split train_data into train_x and train_y
# Use look_back steps to predict the next look_forward steps
train_x, train_y = [], []

for i in range(train_size - look_back - look_forward):
    train_x.append(np.array(train_data[i:i+look_back, :]))
    train_y.append(np.array(train_data[i+look_back:i+look_back+look_forward, :]))
train_x = np.array(train_x)
train_y = np.array(train_y)

test_x, test_y = [], []

for i in range(time_len - train_size - look_back - look_forward):
    test_x.append(np.array(test_data[i:i+look_back, :]))
    test_y.append(np.array(test_data[i+look_back:i+look_back+look_forward, :]))
test_x = np.array(test_x)
test_y = np.array(test_y)

#%%
train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()

#%% 根据id_list筛选feature['link_ID']对应的row组成新的feature_road
feature_road = feature[feature['link_ID'].isin(id_list)]

# 将feature_road按照id_list的顺序进行排序
feature_road = feature_road.set_index('link_ID')
feature_road = feature_road.loc[id_list]
feature_road = feature_road.reset_index()

# feature去掉link_ID列和geometry列
# feature_used = feature_road.drop(columns=['link_ID', 'geometry', 'Kind'])
feature_used = feature_road.drop(columns=['link_ID', 'Kind'])

# Drop rows with NaN values in feature_used
feature_used = feature_used.dropna(axis=1)

# Define a function to implement K-Means algorithm
def kmeans(data, k):
    """
    :param data: 2D numpy array
    :param k: number of clusters
    :return: labels of each data point
    """
    # Initialize centroids randomly
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    # Initialize labels as zeros
    labels = np.zeros(data.shape[0])
    # Iterate until convergence
    while True:
        # Calculate distances between each data point and each centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        # Assign each data point to the closest centroid
        new_labels = np.argmin(distances, axis=0)
        # Check if labels have changed
        if np.array_equal(new_labels, labels):
            break
        # Update labels and centroids
        labels = new_labels
        for i in range(k):
            centroids[i] = np.mean(data[labels == i], axis=0)
    return labels

# Use kmeans function to cluster feature_used into 3 clusters
feature_used = np.array(feature_used.astype('float64'))
# 对feature_used的每一列进行归一化
for i in range(feature_used.shape[1]):
    max_value = np.max(feature_used[:, i])
    feature_used[:, i] = feature_used[:, i] / max_value
# 删除含nan的列
feature_used = feature_used[:, ~np.isnan(feature_used).any(axis=0)]
for dim_value in dims:
    # 对data_flow的每一行的每12列进行求和平均
    data_flow_new = np.zeros((data_flow.shape[0], dim_value))
    for i in range(data_flow.shape[0]):
        for j in range(dim_value):
            data_flow_new[i, j] = np.mean(data_flow[i, j*int(288/dim_value):(j+1)*int(288/dim_value)])
    # 将feature_uesd和data_flow进行拼接
    # feature_used_new = np.concatenate((feature_used, data_flow_new), axis=1)
    feature_used_new = data_flow_new
    feature_used_label = kmeans(feature_used_new, k)
    #%% 存储feature_used_label到.txt文件
    np.savetxt('STC-ML/param/meta/feature_used_label_dim_'+str(look_forward)+'_'+str(k)+'.txt', feature_used_label)

    #%%
    # 读取feature_used_label
    feature_used_label = np.loadtxt('STC-ML/param/meta/feature_used_label_dim_'+str(look_forward)+'_'+str(k)+'.txt')
    # 统计各项出现的次数
    feature_used_label_count = np.zeros(k)
    for i in range(feature_used_label.shape[0]):
        feature_used_label_count[int(feature_used_label[i])] += 1
    print(feature_used_label_count)

    #%%
    def generate_train_x(k):
        train_x_ = {}
        for i in range(k):
            train_x_['train_x_'+str(i)] = []  # 这里可以替换为你的数据生成逻辑
        return train_x_

    def generate_train_y(k):
        train_y_ = {}
        for i in range(k):
            train_y_['train_y_'+str(i)] = []  # 这里可以替换为你的数据生成逻辑
        return train_y_

    train_x_ = generate_train_x(k)
    train_y_ = generate_train_y(k)

    #%%
    for i in range(feature_used_label.shape[0]):
        for j in range(k):
            if feature_used_label[i] == j:
                train_x_['train_x_'+str(j)].append(train_x[:,:,i])
                train_y_['train_y_'+str(j)].append(train_y[:,:,i])

    #%%
    for i in range(k):
        if len(train_x_['train_x_'+str(i)]) > 0:
            train_x_['train_x_'+str(i)] = torch.stack(train_x_['train_x_'+str(i)])
        if len(train_y_['train_y_'+str(i)]) > 0:
            train_y_['train_y_'+str(i)] = torch.stack(train_y_['train_y_'+str(i)])

    #%%
    for i in range(k):
        train_x_['train_x_'+str(i)] = train_x_['train_x_'+str(i)].view(-1, look_back).unsqueeze(-1)
        train_y_['train_y_'+str(i)] = train_y_['train_y_'+str(i)].view(-1, look_forward).unsqueeze(-1)


    #%%
    def generate_test_x(k):
        test_x_ = {}
        for i in range(k):
            test_x_['test_x_'+str(i)] = []  # 这里可以替换为你的数据生成逻辑
        return test_x_

    def generate_test_y(k):
        test_y_ = {}
        for i in range(k):
            test_y_['test_y_'+str(i)] = []  # 这里可以替换为你的数据生成逻辑
        return test_y_

    test_x_ = generate_test_x(k)
    test_y_ = generate_test_y(k)

    #%%
    for i in range(feature_used_label.shape[0]):
        for j in range(k):
            if feature_used_label[i] == j:
                test_x_['test_x_'+str(j)].append(test_x[:,:,i])
                test_y_['test_y_'+str(j)].append(test_y[:,:,i])

    #%%
    for i in range(k):
        if len(test_x_['test_x_'+str(i)]) > 0:
            test_x_['test_x_'+str(i)] = torch.stack(test_x_['test_x_'+str(i)])
        if len(test_y_['test_y_'+str(i)]) > 0:
            test_y_['test_y_'+str(i)] = torch.stack(test_y_['test_y_'+str(i)])

    #%%
    for i in range(k):
        test_x_['test_x_'+str(i)] = test_x_['test_x_'+str(i)].view(-1, look_back).unsqueeze(-1)
        test_y_['test_y_'+str(i)] = test_y_['test_y_'+str(i)].view(-1, look_forward).unsqueeze(-1)

    #%%
    # 创建训练集和测试集的TensorDataset
    train_dataset_ = {}
    test_dataset_ = {}
    for i in range(k):
        train_dataset_['train_dataset_'+str(i)] = TensorDataset(train_x_['train_x_'+str(i)], train_y_['train_y_'+str(i)])
        test_dataset_['test_dataset_'+str(i)] = TensorDataset(test_x_['test_x_'+str(i)], test_y_['test_y_'+str(i)])

    #%%
    # 定义一个批处理大小
    batch_size = 128

    # 创建训练集和测试集的DataLoader
    train_loader_ = {}
    test_loader_ = {}
    for i in range(k):
        train_loader_['train_loader_'+str(i)] = DataLoader(train_dataset_['train_dataset_'+str(i)], batch_size=batch_size, shuffle=True)
        test_loader_['test_loader_'+str(i)] = DataLoader(test_dataset_['test_dataset_'+str(i)], batch_size=batch_size, shuffle=False)

    #%%
    # 定义参数
    input_dim = 1
    hidden_dim = 20
    output_dim = 1

    start_time = time.time()

    # 初始化模型
    model = LSTMModel(input_dim, hidden_dim, output_dim).to(device)
    maml = MAML(model)

    #%%
    # 训练模型
    for i in range(k):
        for data, targets in train_loader_['train_loader_'+str(i)]:
            data, targets = data.to(device), targets.to(device)
            maml.outer_loop(data, targets)

    #%% 存储模型参数
    torch.save(maml.model.state_dict(), 'STC-ML/param/meta/maml_model_dim_'+str(look_forward)+'.pth')

    #%% 测试模型，输出测试集的准确率
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true.flatten()), np.array(y_pred.flatten())
        # Delete elements in y_pred that correspond to the deleted elements in y_true
        y_pred = np.delete(y_pred, np.where(y_true == 0))
        # Delete elements in y_true that are equal to 0
        y_true = y_true[y_true != 0]
        return np.mean(np.abs((y_true - y_pred) / y_true))

    #%%
    preds_list = []
    targets_list = []

    #%% 读取模型参数
    for i in range(k):
        maml.model.load_state_dict(torch.load('STC-ML/param/meta/maml_model_dim_'+str(look_forward)+'.pth'))

        for data, targets in train_loader_['train_loader_'+str(i)]:
            data, targets = data.to(device), targets.to(device)
            maml.inner_loop(data, targets)

        torch.save(maml.model.state_dict(), 'STC-ML/param/meta/maml_model_dim_'+str(look_forward)+'_'+str(i)+'.pth')

        maml.model.load_state_dict(torch.load('STC-ML/param/meta/maml_model_dim_'+str(look_forward)+'_'+str(i)+'.pth'))

        for data, targets in test_loader_['test_loader_'+str(i)]:
            data, targets = data.to(device), targets.to(device)
            preds = model(data)
            preds = preds * max_value_flow
            targets = targets * max_value_flow
            preds = preds.cpu().detach().numpy().flatten()
            targets = targets.cpu().detach().numpy().flatten()
            preds_list = np.append(preds_list, preds)
            targets_list = np.append(targets_list, targets)

    end_time = time.time()

    print('Time: {:.4f}s'.format(end_time - start_time))
    print('Test MAE: {:.4f}'.format(mean_absolute_error(preds_list, targets_list)))
    print('Test RMSE: {:.4f}'.format(np.sqrt(mean_squared_error(preds_list, targets_list))))
    print('Test MSE: {:.4f}'.format(mean_squared_error(preds_list, targets_list)))
    print('Test MAPE: {:.4f}'.format(mean_absolute_percentage_error(preds_list, targets_list)))
    print('Test R2: {:.4f}'.format(r2_score(preds_list, targets_list)))

    #%% 将结果打印到log/meta-LSTM.txt文件中
    f = open('STC-ML/log/meta-LSTM_dim.txt', 'a')
    f.write('dim_value: {:.4f}s\n'.format(dim_value))
    f.write('look_back: {:.4f}s\n'.format(look_back))
    f.write('Time: {:.4f}s\n'.format(end_time - start_time))
    f.write('Test MAE: {:.4f}\n'.format(mean_absolute_error(preds_list, targets_list)))
    f.write('Test RMSE: {:.4f}\n'.format(np.sqrt(mean_squared_error(preds_list, targets_list))))
    f.write('Test MSE: {:.4f}\n'.format(mean_squared_error(preds_list, targets_list)))
    f.write('Test MAPE: {:.4f}\n'.format(mean_absolute_percentage_error(preds_list, targets_list)))
    f.write('Test R2: {:.4f}\n'.format(r2_score(preds_list, targets_list)))


# #%% 绘制前1000个真实值和预测值
# import matplotlib.pyplot as plt
# start_index = 0
# length = 500
# plt.figure(figsize=(12, 6))
# # 加大字体
# plt.rcParams['font.size'] = 10
# # 不要加粗字体
# plt.rcParams['font.weight'] = 'normal'
# # 加粗线条
# plt.rcParams['lines.linewidth'] = 2
# # 每个线条加符号
# plt.rcParams['lines.markersize'] = 7
# markers = "dH^*ov"
# # 纵轴范围0-70
# # plt.ylim(0, 70)
# # index = np.arange(0, 4400 * 30, 4400)
# # plt.plot(preds_list[index], label='Predictions', color='red', linewidth=2, marker=markers[0])
# # plt.plot(targets_list[index], label='Targets', color='grey', linewidth=4, marker=markers[5])
# plt.plot(targets_list[start_index:length], label='Targets', color='grey')
# plt.plot(preds_list[start_index:length], label='Predictions', color='red')
# plt.legend(loc='upper right')
# # plt.xticks(np.arange(0, length+1, 5), np.arange(0, (length+1)*5, 25))
# plt.xlabel('Time Point (min)')
# plt.ylabel('Traffic Flow (km/h)')
# plt.show()










