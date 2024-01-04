import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import torch
from torch.utils.data import DataLoader


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
    data = np.load('dataset/PPI_500_500.npy')
    res = []
    for row in range(len(data)):
        for col in range(row + 1, len(data[0])):
            res.append([row, col, data[row, col]])
    return res


def get_dataset():
    ppi_list = transform_ppi_matrix()
    expression_data = np.load('dataset/RNAseq_feature_500_1089.npy')
    X = []
    y = []
    for i in ppi_list:
        X.append([expression_data[i[0]], expression_data[i[1]]])
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
    pass
