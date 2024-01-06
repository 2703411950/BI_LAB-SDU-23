from cala import *
import numpy as np


def preLabel(cV):
    nCls, nSam = np.shape(cV)

    B, index = iMax(cV, axis=0)
    labels = index + 1

    return labels


def nonneg(Fea):
    nFea = len(Fea)
    for i in range(nFea):
        tmx = Fea[i]
        mVal = np.min(np.min(tmx))
        tmx = tmx - mVal
        Fea[i] = tmx

    return Fea


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This function implements the multiplicative algorithm of NMF
# Reference:
# D. Lee and S. Seung, Algorithms for Non-negative Matrix Factorization,
# NIPS, 2000.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def nmf(X, r, nIter):
    xRow, xCol = np.shape(X)
    W = np.random.rand(xRow, r)
    W = justNorm(W)
    H = np.random.rand(r, xCol)
    H = justNorm(H)

    for ii in range(nIter):
        # +++++ Update H +++++
        tmp = np.dot(np.transpose(W), X)  # r * xCol
        tnp = np.dot(np.transpose(W), W)  # r * r
        tnp = np.dot(tnp, H)  # r * xCol
        tm = tmp / tnp

        H = H * tm  # r * xCol

        # +++++ Update W +++++
        tmp = np.dot(X, np.transpose(H))  # xRow * r
        tnp = np.dot(W, H)  # xRow * xCol
        tnp = np.dot(tnp, np.transpose(H))  # xRow * r

        tm = tmp / tnp
        W = W * tm

        # +++++ Check the objective +++++
        tmp = np.dot(W, H)
        obj = X - tmp
        obj = norm(obj, 1)

        str = 'NMF: The %d-th iteration: ' % ii + '%f' % obj
        print(str)

        if obj < 1e-7:
            break

    return W, H


def totalObj(Fea, U, V, cV, lamda):
    nFea = len(Fea)
    obj = 0

    for i in range(nFea):
        tmx = Fea[i]
        tmu = U[i]
        tmv = V[i]
        tml = lamda[i]

        tmp = np.dot(tmu, tmv)
        tmp = tmx - tmp
        tm = norm(tmp, 1)

        q = np.sum(tmu, axis=0)
        Q = np.diag(q)  # r * r
        tmp = np.dot(Q, tmv)  # r * nCol
        tmp = tmp - cV
        tn = tml * norm(tmp, 1)

        tmn = tm + tn
        obj = obj + tmn

    return obj


def calObj(X, U, V, cV, Q, lamda):
    tmp = np.dot(U, V)
    tmp = X - tmp
    tm = norm(tmp, 1)

    tmp = np.dot(Q, V)  # r * nCol
    tmp = tmp - cV  # r * nCol
    tn = lamda * norm(tmp, 1)

    obj = tm + tn

    return obj


def pervNMF(X, U, V, cV, lamda, maxIter):
    nRow, nCol = np.shape(X)
    _, r = np.shape(U)

    obj = 1e7
    for ii in range(maxIter):
        # +++++ Update U +++++
        tmp = np.dot(X, np.transpose(V))  # nRow * r
        tmq = V * cV  # r * nCol
        tmq = np.sum(tmq, axis=1)  # r * 1
        tmq = repVec(tmq, nRow)  # r * nRow
        tmq = np.transpose(tmq)  # nRow * r

        tm = tmp + lamda * tmq

        tnp = np.dot(U, V)  # nRow * nCol
        tnp = np.dot(tnp, np.transpose(V))  # nRow * r
        tnq = V ** 2
        tnq = np.sum(tnq, axis=1)  # r * 1
        tnq = repVec(tnq, nRow)
        tnq = np.transpose(tnq)  # nRow * r
        tnq = U * tnq  # nRow * r
        tnq = np.sum(tnq, axis=0)  # 1 * r
        tnq = repVec(tnq, nRow)
        tnq = np.transpose(tnq)  # nRow * r

        tn = tnp + lamda * tnq

        tmn = tm / tn
        U = U * tmn

        # +++++ Normalize U and V +++++
        q = np.sum(U, axis=0)  # 1 * r
        Q = np.diag(q)
        tmp = q ** -1
        Qf = np.diag(tmp)

        U = np.dot(U, Qf)  # nRow * r
        V = np.dot(Q, V)  # r * nCol

        # +++++ Update V +++++
        tmp = np.dot(np.transpose(X), U)  # nCol * r
        tmq = np.transpose(cV)  # nCol * r
        tm = tmp + lamda * tmq

        tnp = np.dot(np.transpose(V), np.transpose(U))  # nCol * nRow
        tnp = np.dot(tnp, U)  # nCol * r
        tnq = np.transpose(V)
        tn = tnp + lamda * tnq

        tmn = tm / tn
        tmn = np.transpose(tmn)  # r * nCol
        V = tmn * V  # r * nCol

        # +++++ Check the objective +++++
        oldObj = obj
        obj = calObj(X, U, V, cV, Q, lamda)
        tmp = obj - oldObj
        delta = norm(tmp, 2)
        if delta < 1e-7:
            break

    str = 'The final objective: %f' % obj

    return U, V


def mvnmf(Fea, r, lamda, maxIter):
    """

    :param Fea:所有视图的矩阵构成的列表，每个矩阵是一个视图 [[view1], [view2], ...]，每个view中，一个列是一个样本
    :param r:表示分解后的特征矩阵U和系数矩阵V的秩
    :param lamda:每个视图在共享特征矩阵cV更新时的权重
    :param maxIter:迭代的次数
    :return:每个视图的特征矩阵U：表示每个视图数据的特征表示，可以看作是对原始数据的降维和提取的结果。
            每个视图的系数矩阵V：表示特征矩阵U的系数，用于重构原始数据。
            共享的特征矩阵cV：表示所有视图数据共享的特征，可以理解为不同视图之间共同的特征表示。
    """
    nFea = len(Fea)
    n = len(lamda)
    assert nFea == n, 'The length of features and parameters are not identical !'

    tmx = Fea[0]
    nRow, nCol = np.shape(tmx)

    # +++++ Initialize the Matrices +++++
    U = []
    V = []
    for i in range(nFea):
        tmx = Fea[i]
        tmx = justNorm(tmx)
        Fea[i] = tmx

        tmu, tmv = nmf(tmx, r, 200)
        U.append(tmu)
        V.append(tmv)

    obj = 1e7
    # +++++ Iterative learning +++++
    for ii in range(maxIter):
        cV = np.zeros((r, nCol))
        # +++++ Calcuate cV +++++
        for i in range(nFea):
            tmu = U[i]
            tmv = V[i]
            q = np.sum(tmu, axis=0)
            Q = np.diag(q)

            tmp = np.dot(Q, tmv)  # r * nCol
            tmp = lamda[i] * tmp
            cV = cV + tmp

        tn = np.sum(lamda)
        cV = cV / tn

        # +++++ Update each view +++++
        for i in range(nFea):
            tmx = Fea[i]
            tmu = U[i]
            tmv = V[i]
            tml = lamda[i]

            tmu, tmv = pervNMF(tmx, tmu, tmv, cV, tml, maxIter)
            U[i] = tmu
            V[i] = tmv

        # +++++ Check total objective +++++
        oldObj = obj
        obj = totalObj(Fea, U, V, cV, lamda)
        str = 'Multi-view NMF: The %d-th iteration: ' % ii + '%f' % obj
        print(str)
        with open('loss_100.txt', 'a') as f:
            f.write(f'({ii.__str__()},{obj.__str__()})\n')

        tmp = obj - oldObj
        delta = norm(tmp, 2)
        if delta < 1e-7:
            break

    return U, V, cV


if __name__ == '__main__':
    pass
