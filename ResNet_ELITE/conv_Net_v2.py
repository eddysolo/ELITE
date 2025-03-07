import torch
import torch.nn as nn
import torch.nn.functional as F


def convrelu(in_channels, out_channels, kernel, padding, flag_relu=1):
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
    torch.nn.init.xavier_uniform_(conv_layer.weight)
    relu_layer = nn.ReLU(inplace=True)
    if flag_relu:
        return nn.Sequential(conv_layer, relu_layer,)
    else:
        return nn.Sequential(conv_layer)


class DCEConvNet(nn.Module):
    def __init__(self):
        super(DCEConvNet, self).__init__()
        self.layer1 =convrelu(144,100,(3,3),(1,1)) # input/output/conv/padding
        self.layer2 =convrelu(100,80,(3,3),(1,1))

        self.layer01 = convrelu(80,50,(3,3),(1,1))
        self.layer02 = convrelu(50,20,(3,3),(1,1))
        self.layer03 = convrelu(100,50,(3,3),(1,1))
        self.layer05 = convrelu(50,50,(3,3),(1,1))
        self.layer06 = convrelu(70,100,(3,3),(1,1))

        self.layer3 = convrelu(100,144,(3,3),(1,1))
        self.layer4 = convrelu(144,144,(3,3),(1,1))
        self.layer5 = convrelu(144,144,(3,3),(1,1), flag_relu=0)

    def forward(self, input):
        numBatch = input.shape[0]
        #input = input / torch.max(input)
        x = self.layer1(input)
        lyr2 = self.layer2(x)

        x = self.layer01(lyr2)
        lyr02 = self.layer02(x)
        x = torch.cat([lyr02,lyr2],dim=1) # skipping loop

        x = self.layer03(x)
        x = self.layer05(x)

        x = torch.cat([lyr02,x],dim=1)
        x = self.layer06(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x



if __name__ == "__main__":
    device = 'cpu'
    model = DCEConvNet().to(device)
    out = model(torch.zeros(10,144,160,160).to(device))
    print(out.shape)