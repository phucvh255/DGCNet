import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torchvision.models as models
from libs.models.DualGCNNet import DualGCNHead

SPEEDUP_SCALE = 512
BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

class FCN(nn.Module):

    def __init__(self, bbNet: nn.Module, num_classes):
        super(FCN, self).__init__()
        self.backbone = bbNet
        self.backbone.eval()
 
        self.mergeLayers0 = DummyLayer()

        self.mergeLayers1 = HLayer(2048 + 1024, 1024)
        self.mergeLayers2 = HLayer(1024 + 512, 512)
        self.mergeLayers3 = HLayer(512 + 256, 256)
        
        self.mergeLayers4 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.bn5 = nn.BatchNorm2d(256, momentum=0.003)

        # Output Layer
        self.textScale = 512

        # self.head = DualGCNHead(256, 128, num_classes)

        self.dsn = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.representation = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=True)
        )


    def forward(self, input):

        input = self.__mean_image_subtraction(input)

        # bottom up

        f = self.__foward_backbone(input)

        g = [None] * 4
        h = [None] * 4

        # i = 1
        h[0] = self.mergeLayers0(f[0])
        g[0] = self.__unpool(h[0])

        # i = 2
        h[1] = self.mergeLayers1(g[0], f[1])
        g[1] = self.__unpool(h[1])

        # i = 3
        h[2] = self.mergeLayers2(g[1], f[2])
        g[2] = self.__unpool(h[2])

        # i = 4
        h[3] = self.mergeLayers3(g[2], f[3])
        g[3] = self.__unpool(h[3])
        
       
        # final stage
        final = self.mergeLayers4(g[3])
        final = self.__unpool(final)
        final = self.bn5(final)
        final = F.relu(final)
        if self.training:
            representation = self.representation(final)
        clf = self.dsn(final)
        # prediction = self.dsn(final)
        # final = self.head(final)
        representation = self.representation(final)

        if self.training:
            return [clf, representation]
        else:
            return clf
            

    def __foward_backbone(self, input):
        conv2 = None
        conv3 = None
        conv4 = None
        output = None # n * 7 * 7 * 2048

        for name, layer in self.backbone.named_children():
            input = layer(input)
            if name == 'layer1':
                conv2 = input
            elif name == 'layer2':
                conv3 = input
            elif name == 'layer3':
                conv4 = input
            elif name == 'layer4':
                output = input
                break

        return output, conv4, conv3, conv2

    def __unpool(self, input):
        _, _, H, W = input.shape
        return F.interpolate(input, mode = 'bilinear', scale_factor = 2, align_corners = True)

    def __mean_image_subtraction(self, images, means = [123.68, 116.78, 103.94]):
        '''
        image normalization
        :param images: bs * w * h * channel
        :param means:
        :return:
        '''
        num_channels = images.data.shape[1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')
        for i in range(num_channels):
            images.data[:, i, :, :] -= means[i]

        return images


class DummyLayer(nn.Module):

    def forward(self, input_f):
        return input_f


class HLayer(nn.Module):

    def __init__(self, inputChannels, outputChannels):
        """
        :param inputChannels: channels of g+f
        :param outputChannels:
        """
        super(HLayer, self).__init__()

        self.conv2dOne = nn.Conv2d(inputChannels, outputChannels, kernel_size = 1)
        self.bnOne = nn.BatchNorm2d(outputChannels, momentum=0.003)

        self.conv2dTwo = nn.Conv2d(outputChannels, outputChannels, kernel_size = 3, padding = 1)
        self.bnTwo = nn.BatchNorm2d(outputChannels, momentum=0.003)

    def forward(self, inputPrevG, inputF):
        input = torch.cat([inputPrevG, inputF], dim = 1)
        output = self.conv2dOne(input)
        output = self.bnOne(output)
        output = F.relu(output)

        output = self.conv2dTwo(output)
        output = self.bnTwo(output)
        output = F.relu(output)

        return output

def DualSeg_res50(num_classes=2):
    model = FCN(models.resnet50(pretrained=False), num_classes)
    return model


def DualSeg_res101(num_classes=2):
    model = FCN(models.resnet101(pretrained=True), num_classes)
    return model

if __name__=="__main__":
    model = FCN(models.resnet50(pretrained=True), 2)
    # print(model)