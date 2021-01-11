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
        
        self.mode = 0

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
        
        if self.mode == 1:
            last_dense_data = last_dense.cpu().data.numpy()
            print("last_dense'shape: ", last_dense_data.shape)
            logit_data = logit.cpu().data.numpy()
            print("logit'shape: ", logit_data.shape)
            last_dense_data.tofile("value/last_dense_pytorch_resnet.bin")
            logit_data.tofile("value/logit_pytorch_resnet.bin")
            
        return logit
    
model = MnistResNet().to(device)

summary(model, input_size=(1, 28, 28))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 32
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

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=10000,
                         shuffle=False)

print("data ready")

epochs = 5

for epoch in range(1, epochs + 1):
    for (img, label) in train_loader:
        model.train()
        img, label = img.to(device), label.to(device)
        output = model(img)
        loss = F.nll_loss(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
torch.save(model.state_dict(), 'mnist_resnet.pt')

print("model saved")
print("=======================================================================")

for name, weights in model.named_parameters():
    weights = weights.cpu().data.numpy()
    print(name, ": ", weights.shape)
    weights.tofile(f"weight/{name}_pytorch_resnet.bin")
    
print("=======================================================================")

for test_img, _ in test_loader:
    with torch.no_grad():
        model.eval()
        model.mode = 1
        test_img = test_img.to(device)
        _ = model(test_img) ## to get layer's value
        
print("Finished!")
