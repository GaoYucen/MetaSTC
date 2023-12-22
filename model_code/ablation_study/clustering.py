import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pandas as pd
import numpy as np

import sys
sys.path.append('STC-ML/')

import time

k = 3
look_back = 24

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

# 加载数据并准备训练
data_flow, id_list = load_data('STC-ML/data/traffic_flow/1/20230306/part-00000.pkl')

# 读取feature
feature = pd.read_csv('STC-ML/data/link_feature.csv')

# 确定节点数和时间步数量
# 对data['flow']进行归一化
max_value_flow = np.max(data_flow)
data_flow = data_flow / max_value_flow

# 删除id_list不在feature['link_ID']中的条目，并删除data_flow对应的row
data_flow = data_flow[[i for i,x in enumerate(id_list) if x in feature['link_ID'].values]]
id_list = [i for i in id_list if i in feature['link_ID'].values]

time_len = data_flow.shape[1]
num_nodes = data_flow.shape[0]

#%% 根据id_list筛选feature['link_ID']对应的row组成新的feature_road
feature_road = feature[feature['link_ID'].isin(id_list)]

# 将feature_road按照id_list的顺序进行排序
feature_road = feature_road.set_index('link_ID')
feature_road = feature_road.loc[id_list]
feature_road = feature_road.reset_index()

# feature去掉link_ID列和geometry列
feature_used = feature_road.drop(columns=['link_ID', 'Kind'])

# Drop rows with NaN values in feature_used
feature_used = feature_used.dropna(axis=1)

#%% 聚类
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

def kmeans_pp(data, k):
    """
    :param data: 2D numpy array
    :param k: number of clusters
    :return: labels of each data point
    """
    # Initialize centroids using K-Means++ method
    centroids = np.zeros((k, data.shape[1]))
    centroids[0] = data[np.random.choice(data.shape[0])]
    for i in range(1, k):
        distances = np.min(((data - centroids[:i, np.newaxis])**2).sum(axis=2), axis=0)
        probabilities = distances / np.sum(distances)
        centroids[i] = data[np.random.choice(data.shape[0], p=probabilities)]
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


# 先导入计算Davies-Bouldin Index和Dunn Validity Index的库函数，再测量feature_used_label的聚类优劣
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances
def dunn_index(X, n_clusters, labels):
    # 计算簇的平均距离
    cluster_distances = []
    for cluster in range(n_clusters):
        cluster_points = X[labels == cluster]
        distances = pairwise_distances(cluster_points, squared=True)
        diagonal = np.sort(distances[np.triu_indices(n=distances.shape[0], k=1)])
        cluster_distances.append(diagonal[-1])

    # 计算最大簇间距离
    inter_cluster_distances = pairwise_distances(X, squared=True)
    diagonal = np.sort(inter_cluster_distances[np.triu_indices(n=inter_cluster_distances.shape[0], k=1)])
    max_inter_cluster_distance = diagonal[-1]

    # 计算Dunn指数
    dunn_index = np.max(cluster_distances) / max_inter_cluster_distance

    return dunn_index

#%% 计算feature_used_new特征
feature_used = np.array(feature_used.astype('float64'))
# 对feature_used的每一列进行归一化
for i in range(feature_used.shape[1]):
    max_value_feature = np.max(feature_used[:, i])
    feature_used[:, i] = feature_used[:, i] / max_value_feature
# 删除含nan的列
feature_used = feature_used[:, ~np.isnan(feature_used).any(axis=0)]
# 对data_flow的每一行的每12列进行求和平均
data_flow_new = np.zeros((data_flow.shape[0], 12))
for i in range(data_flow.shape[0]):
    for j in range(12):
        data_flow_new[i, j] = np.mean(data_flow[i, j*look_back:(j+1)*look_back])
#%% 将feature_uesd和data_flow进行拼接
# feature_used_new = np.concatenate((feature_used, data_flow_new), axis=1)
feature_used_new = data_flow_new

#%% 分别使用kmeans和kmeans_pp进行聚类
cluster_method_list = ['kmeans', 'kmeans_pp']
for clutser_method in cluster_method_list:
    start_time = time.time()
    print('Use ', clutser_method, ' to cluster')
    if clutser_method == 'kmeans':
        feature_used_label = kmeans(feature_used_new, k)
    elif clutser_method == 'kmeans_pp':
        feature_used_label = kmeans_pp(feature_used_new, k)
    # 存储feature_used_label到.txt文件
    np.savetxt('STC-ML/param/clustering/flow_feature_used_label_'+clutser_method+'_'+str(k)+'.txt', feature_used_label)

    feature_used_label = np.loadtxt('STC-ML/param/clustering/flow_feature_used_label_'+clutser_method+'_'+str(k)+'.txt')

    dbi = davies_bouldin_score(feature_used_new, feature_used_label)
    print("Davies-Bouldin Index:", dbi)

    # 示例使用
    dunn = dunn_index(feature_used_new, k, feature_used_label)
    print("Dunn Index:", dunn)

    end_time = time.time()

    # 记录Davies-Bouldin Index和Dunn Index
    with open('STC-ML/log/clustering/dbi_dunn_index_'+clutser_method+'_'+str(k)+'.txt', 'w') as f:
        f.write('DBI: '+str(dbi)+'\n')
        f.write('DVI: '+str(dunn)+'\n')
        f.write('Time: '+str(end_time-start_time)+'\n')
        f.close()



