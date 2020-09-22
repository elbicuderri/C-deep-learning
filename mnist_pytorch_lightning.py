import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.nn import Conv2d, MaxPool2d, Linear, BatchNorm2d
from torch.optim import Adam
import pytorch_lightning as pl


class MNISTModel(pl.LightningModule):

    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv = Conv2d(in_channels=1, out_channels=5, kernel_size=(5, 5), padding=(2, 2), bias=True)
        self.maxpool = MaxPool2d(kernel_size=2, stride=(2, 2))
        self.dense1 = Linear(5 * 14 * 14, 120)
        self.dense2 = Linear(120, 10)

    def forward(self, x):
        insize = x.size(0)
        x = x.float()
        conv = self.conv(x)
        maxpool = self.maxpool(conv)
        relu_maxpool = F.relu(maxpool)
        flatten = relu_maxpool.view(insize, -1)
        dense1 = self.dense1(flatten)
        relu_dense1 = F.relu(dense1)
        dense2 = self.dense2(relu_dense1)
        result = F.softmax(dense2, dim=1)
        return result

    def configure_optimizers(self):
        # REQUIRED
        return Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_index):
        # REQUIRED
        x, y = batch
        y_predicted = self(x)
        loss = F.cross_entropy(y_predicted, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_index):
        # OPTIONAL
        x, y = batch
        y_predicted = self(x)
        return {'val_loss': F.cross_entropy(y_predicted, y)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return avg_loss

    def test_step(self, batch, batch_index):
        # OPTIONAL
        x, y = batch
        y_tested = self(x)
        return {'test_loss': F.cross_entropy(y_tested, y)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=100)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32)

mnist_model = MNISTModel()
print(mnist_model)

# most basic trainer, uses good defaults (1 gpu)
trainer = pl.Trainer(gpus=1, max_epochs=3)
trainer.fit(mnist_model)

print('finished')

## https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=HOk9c4_35FKg
## https://github.com/PyTorchLightning/pytorch-lightning