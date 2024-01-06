import os

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mvnmf import nonneg, mvnmf


# 画出误差随聚类类别数量的变化曲线
def elbow(data):
    MSE = []  # 存放每次结果的误差平方和
    for k in range(2, 20):
        km = KMeans(n_clusters=k)  # 构造聚类器
        km.fit(data)
        MSE.append(km.inertia_)
    # inertia_:Sum of squared distances of samples to their closest cluster center.
    # inertia_:样本到其最近聚类中心的平方距离之和
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, 20), MSE, 'o-')
    plt.xticks(range(0, 22, 1))
    plt.grid(linestyle='--')
    plt.xlabel("Number of Clusters Initialized")
    plt.ylabel('SSE')
    plt.show()


def kmeansCluster(X, n_clusters, random_state=0, n_init=10):
    # 创建 KMeans 对象
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    # 对数据进行聚类
    kmeans.fit(X)
    # 获取聚类标签
    labels = kmeans.labels_
    return labels


# 可视化聚类结果
def show(data):
    labels = kmeansCluster(data, 10)  # 一维的ndarry数组，取值为0-9
    tsne = TSNE(n_components=2, random_state=0)
    transformed_data = tsne.fit_transform(data)
    index_list = []
    for i in range(10):
        arr = [(j + 200 * i) for j in range(200)]
        index_list.append(arr)

    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lightcoral', 'tomato', 'gold']
    ind = 0
    for data in index_list:
        plt.scatter(transformed_data[data, 0], transformed_data[data, 1], color=color_list[ind % len(color_list)])
        ind += 1
        print(len(data))
    plt.show()


def load_data(path):
    res = [np.load(path + "/dataset/norm_RNAseq_feature_500_1089.npy").T,
           np.load(path + "/dataset/norm_sequence_feature_500_256.npy").T,
           np.load(path + "/dataset/dense_miRNA.npy").T]
    return res


if __name__ == '__main__':
    root_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    dataset = load_data(root_path)
    dataset = nonneg(dataset)
    U, V, cV = mvnmf(dataset, 1000, [1/3, 1/3, 1/3], 100)
    embedding_v = np.array(cV).T
    np.savetxt(root_path + '/multiview_embedding.txt', embedding_v)
    np.save(root_path + '/multiview_embedding.npy', embedding_v)

    elbow(embedding_v)

    show(embedding_v)
