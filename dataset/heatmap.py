import numpy as np
import torch
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.decomposition import TruncatedSVD

from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")  # 过滤掉警告的意思


# 该函数接受一个二维numpy数组和需要降维到的维度，返回降维后的数据
def pca(data, dimension):
    P = PCA(n_components=dimension)
    transformed_data = P.fit_transform(data)
    return transformed_data


# 只能降维到3维度以下
def tsne(data, dimension):
    tsne = TSNE(n_components=dimension, random_state=0)
    transformed_data = tsne.fit_transform(data)
    return transformed_data


def rowAverage(feaMatrix):
    averages = np.sum(feaMatrix) / 500
    row_sums = feaMatrix.sum(axis=1)
    scalefactor = averages / row_sums
    feaNew = (feaMatrix.T * scalefactor).T
    return feaNew


# SVD分解降维


# 该函数接受一个二维numpy数组，数组的每一列是一个数据点
# 该函数绘制一个热力图，表示数据点之间的关系
def plot_heatmap(data, plot_name):
    # 将data转化为dataframe，并加入列名
    data = pd.DataFrame(data, columns=[i for i in range(data.shape[1])])
    # 图片显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 减号unicode编码
    corr = data.corr()
    # 计算各变量之间的相关系数
    ax = plt.subplots(figsize=(20, 16))  # 调整画布大小, vmin=-0.3, vmax=0.3
    ax = sns.heatmap(corr, square=True, annot=False)  # 画热力图   annot=True 表示显示系数
    # 设置刻度字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(plot_name, fontsize=24)
    plt.show()


from sklearn.preprocessing import StandardScaler


def stand(x):
    # 假设x是一个二维数组，需要按行做标准化
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled


if __name__ == "__main__":
    data2 = np.load('sequence_feature_500_256.npy')
    # non_zero_rows = np.any(data2 != 0, axis=1)
    # data2 = data2[np.ix_(non_zero_rows)]
    #
    # data2 = data2[0:20, :]
    # data2_pca = pca(data2, dimension=15)
    #
    # # 绘制热力图
    # plot_heatmap(data2_pca.T, 'pca')
    # plot_heatmap(data2.T, 'raw')
    # # data.drop(['证券简称','年份'], axis=1, inplace=True) #删除无关的列

    # data2 = np.load("RNAseq_feature_500_1089.npy")

    data2_avg = np.load("norm_sequence_feature_500_256.npy")
    data2 = data2[0:80, :]
    data2_avg = data2_avg[0:80, :]

    plot_heatmap(data2.T, 'RNAseq_feature')
    plot_heatmap(data2_avg.T, 'exchange')
    # data.drop(['证券简称','年份'], axis=1, inplace=True) #删除无关的列
