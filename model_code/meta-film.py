import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn, optim
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time
from scipy import signal
from scipy import special as ss
import torch.nn.functional as F

look_back = 24
look_forward = 6

k = 5

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


def transition(N):
    Q = np.arange(N, dtype=np.float64)
    R = (2 * Q + 1)[:, None]  # / theta
    j, i = np.meshgrid(Q, Q)
    A = np.where(i < j, -1, (-1.) ** (i - j + 1)) * R
    B = (-1.) ** Q[:, None] * R
    return A, B


class HiPPO_LegT(nn.Module):
    def __init__(self, N, dt=1.0, discretization='bilinear'):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super(HiPPO_LegT, self).__init__()
        self.N = N
        A, B = transition(N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)

        self.register_buffer('A', torch.Tensor(A).to(device))
        self.register_buffer('B', torch.Tensor(B).to(device))
        vals = np.arange(0.0, 1.0, dt)
        self.register_buffer('eval_matrix', torch.Tensor(
            ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T).to(device))

    def forward(self, inputs):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """
        c = torch.zeros(inputs.shape[:-1] + tuple([self.N])).to(device)
        cs = []
        for f in inputs.permute([-1, 0, 1]):
            f = f.unsqueeze(-1)
            new = f @ self.B.unsqueeze(0)
            c = F.linear(c, self.A) + new
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, ratio=0.5):
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.modes = min(32, seq_len // 2)
        self.index = list(range(0, self.modes))

        self.scale = (1 / (in_channels * out_channels))
        self.weights_real = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))
        self.weights_imag = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))

    def compl_mul1d(self, order, x, weights_real, weights_imag):
        return torch.complex(torch.einsum(order, x.real, weights_real) - torch.einsum(order, x.imag, weights_imag),
                             torch.einsum(order, x.real, weights_imag) + torch.einsum(order, x.imag, weights_real))

    def forward(self, x):
        B, H, E, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, H, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        a = x_ft[:, :, :, :self.modes]
        out_ft[:, :, :, :self.modes] = self.compl_mul1d("bjix,iox->bjox", a, self.weights_real, self.weights_imag)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2205.08897
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs['task_name']
        self.configs = configs
        # self.modes = configs.modes

        # self.seq_len = configs.seq_len
        # self.label_len = configs.label_len
        # self.pred_len = configs.seq_len if configs.pred_len == 0 else configs.pred_len
        self.seq_len = configs['seq_len']
        self.label_len = configs['label_len']
        self.pred_len = configs['seq_len'] if configs['pred_len'] == 0 else configs['pred_len']

        self.seq_len_all = self.seq_len + self.label_len

        # self.output_attention = configs.output_attention
        # self.layers = configs.e_layers
        # self.enc_in = configs.enc_in
        # self.e_layers = configs.e_layers
        self.output_attention = configs['output_attention']
        self.layers = configs['e_layers']
        self.enc_in = configs['enc_in']
        self.e_layers = configs['e_layers']

        # b, s, f means b, f
        # self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
        # self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))
        self.affine_weight = nn.Parameter(torch.ones(1, 1, configs['enc_in']))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs['enc_in']))

        self.multiscale = [1, 2, 4]
        self.window_size = [256]
        # configs.ratio = 0.5
        configs['ratio'] = 0.5
        self.legts = nn.ModuleList(
            [HiPPO_LegT(N=n, dt=1. / self.pred_len / i) for n in self.window_size for i in self.multiscale])
        self.spec_conv_1 = nn.ModuleList([SpectralConv1d(in_channels=n, out_channels=n,
                                                         seq_len=min(self.pred_len, self.seq_len),
                                                         ratio=configs['ratio']) for n in
                                          self.window_size for _ in range(len(self.multiscale))])
        self.mlp = nn.Linear(len(self.multiscale) * len(self.window_size), 1)

        # if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
        #     self.projection = nn.Linear(
        #         configs.d_model, configs.c_out, bias=True)
        # if self.task_name == 'classification':
        #     self.act = F.gelu
        #     self.dropout = nn.Dropout(configs.dropout)
        #     self.projection = nn.Linear(
        #         configs.enc_in * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec_true, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        x_enc = x_enc * self.affine_weight + self.affine_bias
        x_decs = []
        jump_dist = 0
        for i in range(0, len(self.multiscale) * len(self.window_size)):
            x_in_len = self.multiscale[i % len(self.multiscale)] * self.pred_len
            x_in = x_enc[:, -x_in_len:]
            legt = self.legts[i]
            x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])[:, :, :, jump_dist:]
            out1 = self.spec_conv_1[i](x_in_c)
            if self.seq_len >= self.pred_len:
                x_dec_c = out1.transpose(2, 3)[:, :, self.pred_len - 1 - jump_dist, :]
            else:
                x_dec_c = out1.transpose(2, 3)[:, :, -1, :]
            x_dec = x_dec_c @ legt.eval_matrix[-self.pred_len:, :].T
            x_decs.append(x_dec)
        x_dec = torch.stack(x_decs, dim=-1)
        x_dec = self.mlp(x_dec).squeeze(-1).permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        x_dec = x_dec - self.affine_bias
        x_dec = x_dec / (self.affine_weight + 1e-10)
        x_dec = x_dec * stdev
        x_dec = x_dec + means
        return x_dec

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        x_enc = x_enc * self.affine_weight + self.affine_bias
        x_decs = []
        jump_dist = 0
        for i in range(0, len(self.multiscale) * len(self.window_size)):
            x_in_len = self.multiscale[i % len(self.multiscale)] * self.pred_len
            x_in = x_enc[:, -x_in_len:]
            legt = self.legts[i]
            x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])[:, :, :, jump_dist:]
            out1 = self.spec_conv_1[i](x_in_c)
            if self.seq_len >= self.pred_len:
                x_dec_c = out1.transpose(2, 3)[:, :, self.pred_len - 1 - jump_dist, :]
            else:
                x_dec_c = out1.transpose(2, 3)[:, :, -1, :]
            x_dec = x_dec_c @ legt.eval_matrix[-self.pred_len:, :].T
            x_decs.append(x_dec)
        x_dec = torch.stack(x_decs, dim=-1)
        x_dec = self.mlp(x_dec).squeeze(-1).permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        x_dec = x_dec - self.affine_bias
        x_dec = x_dec / (self.affine_weight + 1e-10)
        x_dec = x_dec * stdev
        x_dec = x_dec + means
        return x_dec

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        x_enc = x_enc * self.affine_weight + self.affine_bias
        x_decs = []
        jump_dist = 0
        for i in range(0, len(self.multiscale) * len(self.window_size)):
            x_in_len = self.multiscale[i % len(self.multiscale)] * self.pred_len
            x_in = x_enc[:, -x_in_len:]
            legt = self.legts[i]
            x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])[:, :, :, jump_dist:]
            out1 = self.spec_conv_1[i](x_in_c)
            if self.seq_len >= self.pred_len:
                x_dec_c = out1.transpose(2, 3)[:, :, self.pred_len - 1 - jump_dist, :]
            else:
                x_dec_c = out1.transpose(2, 3)[:, :, -1, :]
            x_dec = x_dec_c @ legt.eval_matrix[-self.pred_len:, :].T
            x_decs.append(x_dec)
        x_dec = torch.stack(x_decs, dim=-1)
        x_dec = self.mlp(x_dec).squeeze(-1).permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        x_dec = x_dec - self.affine_bias
        x_dec = x_dec / (self.affine_weight + 1e-10)
        x_dec = x_dec * stdev
        x_dec = x_dec + means
        return x_dec

    def classification(self, x_enc, x_mark_enc):
        x_enc = x_enc * self.affine_weight + self.affine_bias
        x_decs = []
        jump_dist = 0
        for i in range(0, len(self.multiscale) * len(self.window_size)):
            x_in_len = self.multiscale[i % len(self.multiscale)] * self.pred_len
            x_in = x_enc[:, -x_in_len:]
            legt = self.legts[i]
            x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])[:, :, :, jump_dist:]
            out1 = self.spec_conv_1[i](x_in_c)
            if self.seq_len >= self.pred_len:
                x_dec_c = out1.transpose(2, 3)[:, :, self.pred_len - 1 - jump_dist, :]
            else:
                x_dec_c = out1.transpose(2, 3)[:, :, -1, :]
            x_dec = x_dec_c @ legt.eval_matrix[-self.pred_len:, :].T
            x_decs.append(x_dec)
        x_dec = torch.stack(x_decs, dim=-1)
        x_dec = self.mlp(x_dec).squeeze(-1).permute(0, 2, 1)

        # Output from Non-stationary Transformer
        output = self.act(x_dec)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None


# Initialize model, loss function, and optimizer
configs = {
    'task_name': 'long_term_forecast',
    'seq_len': 24,
    'label_len': 24,
    'pred_len': 6,
    'output_attention': True,
    'e_layers': 3,
    'enc_in': 1,
    'ratio': 0.5,
    'd_model': 1,
    'c_out': 1,
    'dropout': 0.1,
    'num_class': 1,
    # Add more parameters as needed
}

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, task_kind):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, task_kind)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, task_kind)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, task_kind):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'param/meta-film/checkpoint_mul_'+str(task_kind)+'.pth')  # 这里会存储迄今最优模型的参数
        # torch.save(model, 'finish_model.pkl') # 这里会存储迄今最优的模型
            self.val_loss_min = val_loss

class MAML:
    def __init__(self, model, early_stopping, lr_inner=0.01, lr_outer=0.001, n_inner_loop=1):
        self.model = model
        self.early_stopping = early_stopping
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.n_inner_loop = n_inner_loop
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_outer)

    def inner_loop(self, data, targets):
        for step in range(self.n_inner_loop):
            self.model.train()
            preds = self.model(data, x_mark_enc_value, x_dec_value, x_mark_dec_value)
            loss = nn.MSELoss()(preds, targets)
            self.model.zero_grad()
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            fast_weights = OrderedDict((name, param - self.lr_inner * grad)
                                       for ((name, param), grad) in zip(self.model.named_parameters(), grads))
            for name, param in self.model.named_parameters():
                param.data = fast_weights[name]
            self.early_stopping(loss, self.model, 0)
            if self.early_stopping.early_stop:
                # print("Early stopping")
                break
        return fast_weights

    def outer_loop(self, data, targets):
        fast_weights = self.inner_loop(data, targets)
        preds = self.model(data, x_mark_enc_value, x_dec_value, x_mark_dec_value)
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
sys.path.append('')

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
data_flow, id_list = load_data('data/traffic_flow/1/20230306/part-00000.pkl')

#%% 读取feature
feature = pd.read_csv('data/link_feature.csv')

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
# 对data_flow的每一行的每12列进行求和平均
data_flow_new = np.zeros((data_flow.shape[0], 12))
for i in range(data_flow.shape[0]):
    for j in range(12):
        data_flow_new[i, j] = np.mean(data_flow[i, j*look_back:(j+1)*look_back])
# 将feature_uesd和data_flow进行拼接
feature_used_new = np.concatenate((feature_used, data_flow_new), axis=1)
feature_used_label = kmeans(feature_used_new, k)
#%% 存储feature_used_label到.txt文件
np.savetxt('param/meta/feature_used_label_film_'+str(look_forward)+'_'+str(k)+'.txt', feature_used_label)

#%%
# 读取feature_used_label
feature_used_label = np.loadtxt('param/meta/feature_used_label_film_'+str(look_forward)+'_'+str(k)+'.txt')
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
criterion = nn.MSELoss()
lr_set = 0.01
num_epochs = 50
x_mark_enc_value = 1
x_dec_value = 1
x_mark_dec_value = 1

start_time = time.time()

# 初始化模型
model = Model(configs).to(device)
early_stopping = EarlyStopping(patience=10, verbose=True)
# model = LSTMModel(input_dim, hidden_dim, output_dim).to(device)
maml = MAML(model, early_stopping)

#%%
# 训练模型
for i in range(k):
    for data, targets in train_loader_['train_loader_'+str(i)]:
        data, targets = data.to(device), targets.to(device)
        maml.outer_loop(data, targets)

#%% 存储模型参数
torch.save(maml.model.state_dict(), 'param/meta-film/maml_model_'+str(look_forward)+'.pth')

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
    maml.model.load_state_dict(torch.load('param/meta-film/maml_model_'+str(look_forward)+'.pth'))

    for data, targets in train_loader_['train_loader_'+str(i)]:
        data, targets = data.to(device), targets.to(device)
        maml.inner_loop(data, targets)

    torch.save(maml.model.state_dict(), 'param/meta-film/maml_model_'+str(look_forward)+'_'+str(i)+'.pth')

    maml.model.load_state_dict(torch.load('param/meta-film/maml_model_'+str(look_forward)+'_'+str(i)+'.pth'))

    for data, targets in test_loader_['test_loader_'+str(i)]:
        data, targets = data.to(device), targets.to(device)
        preds = model(data, x_mark_enc_value, x_dec_value, x_mark_dec_value)
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
f = open('log/meta-film.txt', 'a')
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
# length = 30
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
# index = np.arange(0, 7949 * length, 7949)
# plt.plot(preds_list[index], label='Predictions', color='red', linewidth=2, marker=markers[0])
# plt.plot(targets_list[index], label='Targets', color='grey', linewidth=4, marker=markers[5])
# # plt.plot(targets_list[start_index:length], label='Targets', color='grey')
# # plt.plot(preds_list[start_index:length], label='Predictions', color='red')
# plt.legend(loc='upper right')
# # plt.xticks(np.arange(0, length+1, 5), np.arange(0, (length+1)*5, 25))
# plt.xlabel('Time Point (min)')
# plt.ylabel('Traffic Flow (km/h)')
# plt.savefig('graph/MetaSTC.pdf')
#
# #%% 存储preds_list[index]的数据到npy
# # 将数据转换为NumPy数组
# data = np.array(preds_list[index])
#
# # 保存数据到文件
# np.save('data/MetaSTC_FiLM.npy', data)










