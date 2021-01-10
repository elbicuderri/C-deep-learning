import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MnistResNet(nn.Module):
    def __init__(self):
        super(MnistResNet, self).__init__()

        self.relu = nn.ReLU()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(8)
        )
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(8),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(8),
        )

        self.avg_pool = nn.AvgPool2d(2, stride=2)
        
        self.fc = nn.Linear(8 * 7 * 7, 10)
        
    def forward(self, x):
        batch = x.size(0)
        out0 = self.conv0(x)

        res11 = self.conv1(out0)
        out11 = self.block1(out0)
        out11 += res11
        out11 = self.relu(out11)

        res12 = out11
        out12 = self.block2(out11)
        out12 += res12
        out12 = self.relu(out12)

        out2 = self.avg_pool(out12)
        out2 = out2.view(batch, -1)
        last_dense = self.fc(out2)
        logit = F.log_softmax(last_dense, dim=1)
                    
        return logit


model = MnistResNet().to(device)

summary(model, input_size=(1, 28, 28))

model.load_state_dict(torch.load("mnist_resnet.pt"))

model.conv0._modules['1'].running_mean.cpu().data.numpy().tofile("weight/conv0.bn.mean_pytorch_resnet.bin")
model.conv0._modules['1'].running_var.cpu().data.numpy().tofile("weight/conv0.bn.var_pytorch_resnet.bin")

model.conv1._modules['1'].running_mean.cpu().data.numpy().tofile("weight/conv1.bn.mean_pytorch_resnet.bin")
model.conv1._modules['1'].running_var.cpu().data.numpy().tofile("weight/conv1.bn.var_pytorch_resnet.bin")

model.block1._modules['1'].running_mean.cpu().data.numpy().tofile("weight/block1.bn1.mean_pytorch_resnet.bin")
model.block1._modules['1'].running_var.cpu().data.numpy().tofile("weight/block1.bn1.var_pytorch_resnet.bin")

model.block1._modules['4'].running_mean.cpu().data.numpy().tofile("weight/block1.bn2.mean_pytorch_resnet.bin")
model.block1._modules['4'].running_var.cpu().data.numpy().tofile("weight/block1.bn2.var_pytorch_resnet.bin")

model.block2._modules['1'].running_mean.cpu().data.numpy().tofile("weight/block2.bn1.mean_pytorch_resnet.bin")
model.block2._modules['1'].running_var.cpu().data.numpy().tofile("weight/block2.bn1.var_pytorch_resnet.bin")

model.block2._modules['4'].running_mean.cpu().data.numpy().tofile("weight/block2.bn2.mean_pytorch_resnet.bin")
model.block2._modules['4'].running_var.cpu().data.numpy().tofile("weight/block2.bn2.var_pytorch_resnet.bin")

for name, weight in model.named_parameters():
    weight = weight.cpu().data.numpy()
    print(name, weight.shape)
    weight.tofile(f"weight/{name}_pytorch_resnet.bin") 

print("finished")