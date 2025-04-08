import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np


class CIFAR10_LeNet(nn.Module):
    def __init__(self):
        super(CIFAR10_LeNet, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-4, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-4, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-4, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn2d1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn2d2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn2d3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def summary(self):
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum(np.prod(p.size()) for p in net_parameters)
        self.logger.info("Trainable parameters: {}".format(params))
        self.logger.info(self)


class CIFAR10_LeNet_Autoencoder(nn.Module):
    def __init__(self):
        super(CIFAR10_LeNet_Autoencoder, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)
        # Encoder
        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-4, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-4, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-4, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-4, affine=False)
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(
            int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2
        )
        nn.init.xavier_uniform_(self.deconv1.weight)
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-4, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight)
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-4, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight)
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-4, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight)

    def forward(self, x):
        # Encoder
        x = self.pool(F.leaky_relu(self.bn2d1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn2d2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn2d3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.bn1d(self.fc1(x))
        # Decoder
        x = x.view(x.size(0), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x

    def summary(self):
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum(np.prod(p.size()) for p in net_parameters)
        self.logger.info("Trainable parameters: {}".format(params))
        self.logger.info(self)
