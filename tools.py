import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, recall_score
import ipdb


class Trainer:
    def __init__(self, model, data, decay=4, lr=1e-1, lr_decay_rate=0.5, BATCH_SIZE=256, with_test_flag=False,
                 test_data=None, result_path="result/method1/"):
        self.model = model
        self.decay = decay
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.batch_size = BATCH_SIZE
        self.EPOCH = self.decay * 4 + 1
        self.data = data
        self.result_path = result_path
        self.train_loader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.device = ['gpu' if torch.cuda.is_available() else 'cpu']
        if self.device == 'gpu':
            self.criterion.cuda()
            self.model.to(self.device)

        self.with_test_flag = with_test_flag
        if self.with_test_flag:
            self.tester = Tester(self.model, test_data)

    def train(self):
        self.model.train()
        for epoch in range(self.EPOCH):
            for batch_id, input_data in enumerate(self.train_loader):
                batch_start = time.time()
                self.optimizer.zero_grad()
                x1, x2, y = input_data
                if self.device == 'gpu':
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    y = y.cuda()

                outputs = self.model(x1, x2)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                batch_end = time.time()

                if batch_id % 10 == 0:
                    log_str = f"EPOCH: {epoch + 1}/{self.EPOCH} | batch: {batch_id + 1}/{len(self.train_loader)} " \
                              f"| loss: {loss} | time: {round((batch_end - batch_start), 4)}\n"
                    sys.stdout.write(log_str)
                    write_log(self.result_path + "train_record.txt", log_str)

            if self.with_test_flag:
                self.tester.test()


class Tester:
    def __init__(self, model, data, log_interval=10, BATCH_SIZE=256, result_path="result/method1/"):
        self.model = model
        self.data = data
        self.log_interval = log_interval
        self.batch_size = BATCH_SIZE
        self.test_loader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.result_path = result_path
        self.device = ['gpu' if torch.cuda.is_available() else 'cpu']
        if self.device == 'gpu':
            self.model.to(self.device)

    def test(self):
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, input_data in enumerate(self.test_loader):
                x1, x2, y = input_data
                if self.device == 'gpu':
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    y = y.cuda()

                outputs = self.model(x1, x2)
                # ipdb.set_trace()

                prediction = torch.max(outputs, dim=1)[1]
                y_score = outputs[:, 1]
                pred_y = prediction.data.numpy().squeeze()
                target_y = y.data.numpy()
                acc, recall, auc_score, aupr, f1 = compute_metrics(pred_y, target_y, y_score)

                # print test log
                log_str = f"test: batch: {batch_idx + 1}/{len(self.test_loader)} | acc: {acc} | " \
                          f"recall: {recall} | auc: {auc_score} | f1: {f1}\n"
                sys.stdout.write(log_str)
                write_log(self.result_path + "test_record.txt", log_str)

                # record the total accuracy
                total += y.size(0)
                correct += (pred_y == target_y).sum().item()
        print("Accuracy of the trained network over test set is {:.3f}%".format(correct / total * 100))


def compute_metrics(pred, target, y_score):
    """
    compute total accuracy and the accuracy to predict 1
    """
    indices_1 = np.where(target == 1)
    pred_1 = pred[indices_1]
    # precision = sum(pred_1) / len(indices_1[0])
    # accuracy_total = sum(pred == target) / count
    accuracy = accuracy_score(target, pred)

    # 计算AUROC
    auc_score = roc_auc_score(target, pred)

    # 计算AUPR
    precision, recall, _ = precision_recall_curve(target, y_score)
    aupr = auc(recall, precision)

    # 计算f1-score
    f1 = f1_score(target, pred, average='macro')

    recall = recall_score(target, pred)

    return accuracy, recall, auc_score, aupr, f1


def write_log(filename, string):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(string)


def normalize_rows(arr):
    """
    对NumPy数组按行进行最大最小归一化
    """
    # 计算每一行的最小值和最大值
    min_values = np.min(arr, axis=1)
    max_values = np.max(arr, axis=1)

    # 计算每一行的范围（最大值减最小值）
    range_values = max_values - min_values

    # 按行进行最大最小归一化
    normalized_arr = (arr - min_values[:, np.newaxis]) / range_values[:, np.newaxis]

    return normalized_arr
