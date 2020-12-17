import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from auto_chkpt import registry, SimpleSaver

DEVICE = torch.device("cuda")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
  
    def forward(self, x)->torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def get_mnist(shuffle=True):
    from torch.utils.data import DataLoader
    trainset = DataLoader(datasets.MNIST("dataset", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])), shuffle=shuffle, batch_size=100, num_workers=4)
    testset = DataLoader(datasets.MNIST("dataset", train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])), shuffle=shuffle, batch_size=100, num_workers=4)
    return trainset, testset

def accuracy(model, testset):
    items = 0
    acc = 0
    for data, labels in testset:
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        pred = model(data)
        items += labels.shape[0]
        acc += pred.argmax(1).eq(labels).sum().item()
  
    return acc/items

@registry.watch_training_process()
def train():
    trainset, testset = get_mnist()
    criterion = nn.CrossEntropyLoss()

    net = Net().to(DEVICE)
    optimizer = O.SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    saver = SimpleSaver("checkpoint", "mnist", net, optimizer, device="cuda", chkpt_steps=10, resume=True, memory_snapshot=False)
    registry.attach_saver(saver, "saver_mnist")

    for e in range(saver.epoch, 50):
        for data, labels in trainset:
            net.train()
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            pred = net(data)
            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        acc = accuracy(net, testset)
        print(f"E: {e}, A: {acc:.4f}, S: {saver.epoch}, R: {torch.rand(5)}")

        saver.step()


    return net

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train()