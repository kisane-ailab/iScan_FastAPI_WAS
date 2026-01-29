import torch
import torch.nn as nn
import torch.nn.functional as F
from classification.model.resnet import *

class ClsNet3(nn.Module):
    def __init__(self, num_classes, net='resnet34'):
        super(ClsNet3, self).__init__()
        if net == 'resnet34':
            self.encoder = resnet34()
        elif net == 'resnet18':
            self.encoder = resnet18()
            
        self.linear1 = nn.Linear(512 * 3, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear_softmax = nn.Linear(512, num_classes)
        
    def forward(self, x1, x2, x3):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x3 = self.encoder(x3)
        
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear_softmax(x)
        
        return x

class ClsNet1(nn.Module):
    def __init__(self, num_classes, net='resnet34'):
        super(ClsNet1, self).__init__()
        if net == 'resnet34':
            self.encoder = resnet34()
            
        self.linear1 = nn.Linear(512 * 1, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear_softmax = nn.Linear(512, num_classes)
        
    def forward(self, x1):
        x1 = self.encoder(x1)
        
        ## x = torch.cat([x1], dim=1)
        x = self.linear1(x1)
        x = self.linear2(x)
        x = self.linear_softmax(x)
        
        return x