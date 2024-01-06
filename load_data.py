import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import torch
from torch.utils.data import DataLoader
from tools.training_tools import normalize_rows
from sklearn.decomposition import PCA


def plot_data_distribution(data, n_pic):
    fig, axes = plt.subplots(nrows=int(n_pic/3), ncols=3)

    for i in range(n_pic):
        row_pos = int(i / 3)
        col_pos = i % 3
        axes[row_pos, col_pos].hist(data[i])
        axes[row_pos, col_pos].set_title('mean:{:.2f}, var:{:.2f}'.format(data[i].mean(), data[i].var()))
    plt.tight_layout()
    plt.show()


def draw_heatmap():
    data = np.load('dataset/RNAseq_feature_500_1089.npy')
    data = pd.DataFrame(data)
    a = data.corr()
    print('皮尔逊系数')
    print(a)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.heatmap(data.corr(method='pearson'), linewidths=0.1, vmax=1.0, square=True, linecolor='white', annot=True)
    plt.title('Heatmap')
    plt.show()


def show_ppi_matrix(data):
    G = nx.Graph()
    for i in range(len(data)):
        for j in range(len(data)):
            if data[i][j] == 1:
                G.add_edge(i, j)

    nx.draw(G)
    plt.show()


def transform_ppi_matrix():
    """
    根据互作矩阵，获得蛋白质之间的两两关系
    并且按照所给的正负样本比例划分
    return: numpy数组 [[id1, id2, relationship], ...]
    """
    data = np.load('dataset/PPI_500_500.npy')
    rows, cols = np.triu_indices(len(data), k=1)  # 获取上三角部分的索引
    ppi_list = np.column_stack((rows, cols, data[rows, cols]))

    # 分出正负样本的总数据集
    pos_ids = ppi_list[:, 2] == 1
    neg_ids = ppi_list[:, 2] == 0

    pos_data = ppi_list[pos_ids]
    neg_data = ppi_list[neg_ids]

    # 计算大小
    sample_size = min(len(pos_data), len(neg_data))
    # 分别从正负样本的集合中随机抽样
    pos_samples = np.random.choice(len(pos_data), size=sample_size, replace=False)
    neg_samples = np.random.choice(len(neg_data), size=sample_size, replace=False)

    # 根据抽样的结果取数据
    pos_data = pos_data[pos_samples]
    neg_data = neg_data[neg_samples]
    res = np.concatenate((neg_data, pos_data), axis=0)
    return res


def get_dense():
    miRNA_data = np.load("dataset/miRNA_feature_500_1602.npy")
    n_components = 500
    pca = PCA(n_components)
    dense_feat = pca.fit_transform(miRNA_data)
    dense_feat = np.array(dense_feat)
    np.save("dataset/dense_miRNA.npy", dense_feat)
    return dense_feat


def norm_data():
    seq_data = np.load("dataset/sequence_feature_500_256.npy")
    seq_data = normalize_rows(seq_data)
    np.save("dataset/norm_sequence_feature_500_256.npy", seq_data)

    expression_data = np.load('dataset/RNAseq_feature_500_1089.npy')
    expression_data = normalize_rows(expression_data)
    np.save("dataset/norm_RNAseq_feature_500_1089.npy", expression_data)


def get_dataset():
    ppi_list = transform_ppi_matrix()
    expression_data = np.load('dataset/norm_RNAseq_feature_500_1089.npy')
    seq_data = np.load("dataset/norm_sequence_feature_500_256.npy")
    miRNA_data = np.load("dataset/dense_miRNA.npy")
    # feature dim = 1089+256+500 = 1845
    concatenated = np.concatenate((expression_data, seq_data, miRNA_data), axis=1)
    X = []
    y = []
    for i in ppi_list:
        X.append([concatenated[i[0]], concatenated[i[1]]])
        y.append(i[2])
    return X, y


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        self.X, self.y = get_dataset()

    def __getitem__(self, index):
        return self.X[index][0], self.X[index][1], self.y[index]

    def __len__(self):
        return len(self.y)


if __name__ == '__main__':
    exp_data = np.load("dataset/RNAseq_feature_500_1089.npy")
    plot_data_distribution(exp_data, 6)
