import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, dropout=0.8, n_inputs=1089, n_hidden1=1000, n_hidden2=500, n_outputs=2):
        super(MyModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fc1 = nn.Sequential(
            nn.Linear(2 * n_inputs, n_hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(n_hidden2, n_outputs),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = x.type(torch.float32)
        y = self.fc1(x)
        return y
