import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from tools.training_tools import get_edge_index
import numpy as np


class Classify(nn.Module):
    def __init__(self, dropout=0.3, n_inputs=200, n_outputs=2):
        super(Classify, self).__init__()
        self.linear1 = nn.Linear(n_inputs, 250)
        self.linear2 = nn.Linear(n_inputs, 250)
        self.fc1 = nn.Sequential(
            nn.Linear(500, 1000),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 600),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.BatchNorm1d(600),
            nn.Linear(600, n_outputs),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        x1 = x1.type(torch.float32)
        x2 = x2.type(torch.float32)

        x1 = self.linear1(x1)
        x2 = self.linear2(x2)

        x = torch.cat((x1, x2), dim=1)
        y = self.fc1(x)
        return y


class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 400
        self.in_head = 8
        self.out_head = 1

        self.conv1 = GATConv(200, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid * self.in_head, 200, concat=False,
                             heads=self.out_head, dropout=0.6)

    def forward(self, input_data, adj):
        edge_index = get_edge_index(adj)

        x = F.dropout(input_data, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class MyModel(nn.Module):
    def __init__(self, dropout=0.3, n_inputs=200, n_outputs=2):
        super(MyModel, self).__init__()
        self.GAT = GAT()
        self.classify = Classify()
        self.dataset = np.load("dataset/multiview/multiview_embedding_200.npy")
        self.dataset = torch.from_numpy(self.dataset).type(torch.float32)
        self.adj_mat = np.load("dataset/PPI_500_500.npy")
        self.adj_mat = torch.from_numpy(self.adj_mat).type(torch.float32)

    def forward(self, x1_id, x2_id):
        h_p = self.GAT(self.dataset, self.adj_mat)
        x1 = h_p[x1_id].type(torch.float32)
        x2 = h_p[x2_id].type(torch.float32)
        out = self.classify(x1, x2)
        return out
