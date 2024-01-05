import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, dropout=0.3, n_inputs=256, n_outputs=2):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(n_inputs, 200)
        self.linear2 = nn.Linear(n_inputs, 200)
        self.fc1 = nn.Sequential(
            nn.Linear(400, 1000),
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
