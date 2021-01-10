import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchsummary import summary
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class mnist_model(nn.Module):
    def __init__(self):
        super(mnist_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=1, stride=2, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(5, eps=0.001)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3, padding=1, stride=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(10, eps=0.001)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.dense1 = nn.Linear(10 * 7 * 7, 60)
        self.dense2 = nn.Linear(60, 10)
        self.relu = nn.ReLU()
        self.mode = 0

    def forward(self, x):
        batch = x.size(0)
        conv1 = self.conv1(x) # (5, 14, 14)
        batchnorm1 = self.batchnorm1(conv1)
        relu1 = self.relu(batchnorm1)
        conv2 = self.conv2(relu1) # (10, 14, 14)
        batchnorm2 = self.batchnorm2(conv2)
        relu2 = self.relu(batchnorm2)
        maxpool = self.maxpool(relu2)
        flatten = maxpool.view(batch, -1)
        dense1 = self.dense1(flatten)
        relu_dense1 = self.relu(dense1) 
        dense2 = self.dense2(relu_dense1)
        result = F.log_softmax(dense2, dim=1)
               
        if self.mode == 1:
            def save_value(value, name):
                value_arr = value.cpu().data.numpy()
                print(name, ": ", value_arr.shape)
                value_arr.tofile(f"value/{name}_pytorch_v3.bin")

            value_list = [dense2, result]
            name_list = ["dense2", "result"]
            for v, n in zip(value_list, name_list):
                save_value(v, n)

        return result

model = mnist_model().to(device)

summary(model, input_size=(1, 28, 28))
# print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 100
transform = ToTensor()

#image to Tensor -> image, label
train_dataset = MNIST('../mnist_data/',
                               download=True,
                               train=True,
                               transform=transform)

test_dataset = MNIST("../mnist_data/",
                              train=False,
                              download=True,
                              transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)

print("data ready")

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        # get the index of the max log-probability
        pred = output.argmax(1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(valid_loader.dataset)
    print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))


mean1_list = []
var1_list = []
mean2_list = []
var2_list = []

for epoch in range(0, 5):
    train(epoch)
    mean1 = model.batchnorm1.running_mean.clone()
    variance1 = model.batchnorm1.running_var.clone()
    mean2 = model.batchnorm2.running_mean.clone()
    variance2 = model.batchnorm2.running_var.clone()
    
    test()
    mean1_list.append(mean1)
    var1_list.append(variance1)
    mean2_list.append(mean2)
    var2_list.append(variance2)
    
# print(mean1_list)
# print(var1_list)
# print(mean2_list)
# print(var2_list)

(mean1_list[-1].cpu().data.numpy()).tofile("weight/batchnorm1.mean_pytorch_v3.bin")
(var1_list[-1].cpu().data.numpy()).tofile("weight/batchnorm1.variance_pytorch_v3.bin")
(mean2_list[-1].cpu().data.numpy()).tofile("weight/batchnorm2.mean_pytorch_v3.bin")
(var2_list[-1].cpu().data.numpy()).tofile("weight/batchnorm2.variance_pytorch_v3.bin")

print("=======================================================================")

def calculate():
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            model.mode = 1
            model.eval()
            _ = model(data)

calculate()

print("=======================================================================")

def save_weights(weights, name):
    weights = weights.cpu().data.numpy()
    print(name, ": ", weights.shape)
    weights.tofile(f"weight/{name}_pytorch_v3.bin")
    
for n, w in model.named_parameters():
    save_weights(w, n)

print("Finished!")