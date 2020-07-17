import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

batch_size = 100

transform = transforms.ToTensor() # 0 ~ 1

train_dataset = datasets.MNIST('../mnist_data/',
                               download=True,
                               train=True,
                               transform=transform) # image to Tensor # image, label

test_dataset = datasets.MNIST("../mnist_data/",
                              train=False,
                              download=True,
                              transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 5, kernel_size=5, padding=2, bias=False)
        self.batchnorm = nn.BatchNorm2d(5, eps=0.001)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.dense1 = nn.Linear(5 * 14 * 14, 120)
        self.dense2 = nn.Linear(120, 10)
        self.dump = 0

    def forward(self, x):
        insize = x.size(0)
        x = x.float()
        conv1 = self.conv(x)
        batchnorm = self.batchnorm(conv1)
        maxpool = self.maxpool(batchnorm)
        relu_maxpool = F.relu(maxpool)
        flatten = relu_maxpool.view(insize, -1)
        dense1 = self.dense1(flatten)
        relu_dense1 = F.relu(dense1)
        dense2 = self.dense2(relu_dense1)
        y = F.softmax(dense2, dim=1)
        if self.dump == 1:
            conv1.cpu().data.numpy().tofile("weights_torch/conv_torch.bin")
            batchnorm.cpu().data.numpy().tofile("weights_torch/batchnorm_torch.bin")
            maxpool.cpu().data.numpy().tofile("weights_torch/maxpool_torch.bin")
            relu_maxpool.cpu().data.numpy().tofile("weights_torch/relu_maxpool_torch.bin")
            flatten.cpu().data.numpy().tofile("weights_torch/flatten_torch.bin")
            dense1.cpu().data.numpy().tofile("weights_torch/dense1_torch.bin")
            relu_dense1.cpu().data.numpy().tofile("weights_torch/relu_dense1_torch.bin")
            dense2.cpu().data.numpy().tofile("weights_torch/dense2_torch.bin")
            y.cpu().data.numpy().tofile("weights_torch/result_torch.bin")
        return y

model = Net()

parameters_zero = list(model.parameters())

# print("params_zero: ", parameters_zero)

optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = Variable(data), Variable(target)
        output = model(data)
        #print(output, "\n\n")
        # sum up batch loss
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        # get the index of the max log-probability
        pred = output.argmax(1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(valid_loader.dataset)
    print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

mean_list = []
var_list = []

for epoch in range(0, 5):
    train(epoch)
    test()
    mean = model.batchnorm.running_mean
    variance = model.batchnorm.running_var
    mean_list.append(mean)
    var_list.append(variance)
    print(mean)
    print(variance)

def save_weights(weights, name):
    weights = weights.detach().numpy()
    weights.tofile(f"weights_torch/{name}_torch.bin")

parameters = list(model.parameters())

# mean = model.batchnorm.running_mean
#
# variance = model.batchnorm.running_var
#
# print(mean)
#
# print(variance)

name_list = ["kernel", "gamma", "beta", "W2", "b2", "W3", "b3"]

for w, n in zip(parameters, name_list):
    save_weights(w, n)

save_weights(mean_list[-1], "mean")
save_weights(var_list[-1], "variance")

# name_list = list(model.state_dict())
# print(name_list)

def calculate():
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
    model.dump = 1
    model.eval()
    output = model(data)

calculate()

print("Finished!")

# torch.save(model, "model/mnist_torch.pt")
#
# # 모델의 state_dict 출력
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# 옵티마이저의 state_dict 출력
#print("Optimizer's state_dict:")
#for var_name in optimizer.state_dict():
    #print(var_name, "\t", optimizer.state_dict()[var_name])

# state_dict = model.state_dict()
# for k in state_dict.keys():
#     print(k,":", state_dict[k],"\n")

# for data, target in test_loader:
#     data, target = Variable(data), Variable(target)
#     result = model(data)
#     print(result.shape)
#     # for i in range(20):
#     #     print(result[i, :], target[i])
#     result = result.detach().numpy()
#     print(result.shape)
#     result.tofile("weights_torch/result_torch.bin")
