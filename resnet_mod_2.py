import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ResNetBinary(nn.Module):
    def __init__(self, block, layers, sample_input_D, sample_input_H, sample_input_W, num_classes=2, shortcut_type='B', no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNetBinary, self).__init__()

        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        # Aggiungere un AdaptiveAvgPool3d per ridurre la dimensione spaziale
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # Aggiungi un fully connected layer per la classificazione binaria
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # Cambia il numero di uscite a 2 per la classificazione binaria

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global Average Pooling
        x = self.avgpool(x)  # Produce a (batch_size, 512*block.expansion, 1, 1, 1) tensor

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layer for binary classification
        x = self.fc(x)

        # Applica la funzione sigmoid per ottenere una probabilità tra 0 e 1
        x = torch.sigmoid(x)

        return x
