import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class mnist_model(nn.Module):
    def __init__(self):
        super(mnist_model, self).__init__()

        self.relu = nn.ReLu()
        self.conv = nn.Conv2d(1, 5, kernel_size=5, padding=2, bias=False)
        self.batchnorm = nn.BatchNorm2d(5, eps=0.001)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.dense1 = nn.Linear(5 * 14 * 14, 120)
        self.dense2 = nn.Linear(120, 10)

    def forward(self, x):
        batch = x.size(0)
        conv = self.conv(x)
        batchnorm = self.batchnorm(conv)
        maxpool = self.maxpool(batchnorm)
        relu_maxpool = self.relu(maxpool)
        flatten = relu_maxpool.view(batch, -1)
        dense1 = self.dense1(flatten)
        relu_dense1 = self.relu(dense1)
        dense2 = self.dense2(relu_dense1)
        result = F.softmax(dense2, dim=1)

        return result

model = mnist_model().to(device)
