import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torchvision.models as models

SPEEDUP_SCALE = 512
from torch.nn import BatchNorm2d


import torch
import torch.nn.functional as F
import torch.nn as nn
from libs.models.GALDNet import Bottleneck, conv3x3

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d


class SpatialGCN(nn.Module):
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 BatchNorm2d(plane))

    def forward(self, x):
        # b, c, h, w = x.size()
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        # A = k * q
        # AV = k * q * v
        # AVW = k *(q *v) * w
        AV = torch.bmm(node_q,node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        out = F.relu_(self.out(AVW) + x)
        return out


class DualGCN(nn.Module):
    """
        Feature GCN with coordinate GCN
    """
    def __init__(self, planes, ratio=4):
        super(DualGCN, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = BatchNorm2d(planes // ratio)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = BatchNorm1d(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(planes // ratio * 2)

        #  last fc
        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)

        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes))
        self.gcn_local_attention = SpatialGCN(planes)

        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
                                   BatchNorm2d(planes))

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, feat):
        # # # # Local # # # #
        x = feat
        local = self.local(feat)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = x * local + x

        # # # # Projection Space # # # #
        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x+y)

        # cat or sum, nearly the same results
        out = self.final(torch.cat((spatial_local_feat, g_out), 1))

        return out


class DualGCNHead(nn.Module):
    def __init__(self, inplanes, interplanes, num_classes):
        super(DualGCNHead, self).__init__()
        self.conva = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))
        self.dualgcn = DualGCN(interplanes)
        self.convb = nn.Sequential(nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes + interplanes, interplanes, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        output = self.conva(x)
        output = self.dualgcn(output)
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x, output], 1))
        return output

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

        self.head = DualGCNHead(256, 128, num_classes)

        self.dsn = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
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
            final_dsn = self.dsn(final)
        final = self.head(final)
    
        if self.training:
            return [final, final_dsn]
        else:
            return final    

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
    model = FCN(models.resnet50(pretrained=True), num_classes)
    return model
if __name__=="__main__":
    print("ok")