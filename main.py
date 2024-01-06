from load_data import MyDataset
from model import MyModel
import torch
from tools.training_tools import Trainer


if __name__ == '__main__':
    my_dataset = MyDataset()
    train_size = int(len(my_dataset) * 0.8)
    test_size = len(my_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [train_size, test_size])

    my_train = Trainer(MyModel(), train_dataset, with_test_flag=True, test_data=test_dataset)
    my_train.train()
