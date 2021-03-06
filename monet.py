import torch
import torch.nn as nn


class SingleLayer(nn.Module):
    def __init__(self, nChannels):
        super(SingleLayer, self).__init__()
        self.conv1 = nn.Conv2d(nChannels, nChannels, kernel_size=3,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.acti = nn.ReLU(inplace = True)
    def forward(self, x):
        out = self.bn1(self.acti(self.conv1(x)))
        return out


class MoireCNN(nn.Module):
    def __init__(self, output_all=False, model_path=None):
        super(MoireCNN, self).__init__()

        #self.data_dict = self.load_model(model_path)
        self.output_all = output_all

        self.scale1 = nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        )

        self.scale2 = nn.Sequential(
        nn.Conv2d(32, 32, 3, 2, 1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 3, 1, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        )

        self.scale3 = nn.Sequential(
        nn.Conv2d(64, 64, 3, 2, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
        )


        # #Multi-player
        # self.multiplayer_32 = nn.Sequential(
        #       nn.Conv2d(32, 64, 3, 1, 1, bias=False),
        #       nn.BatchNorm2d(64),
        #       nn.ReLU(True)
        #   )
        # self.multiplayer_64 = nn.Sequential(
        #     nn.Conv2d(64, 64, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True)
        # )

        self.multiple1 = self._make_multiple(32, 5)
        self.multiple2 = self._make_multiple(64, 5)
        self.multiple3 = self._make_multiple(64, 5)
        self.multiple4 = self._make_multiple(64, 5)
        self.multiple5 = self._make_multiple(64, 5)

        #Dowunsample

        self.descale1 = nn.Sequential(
              nn.Conv2d(32, 3, 3, 1, 1, bias=False),
              nn.BatchNorm2d(3),
              nn.ReLU(True)
          )
        self.descale2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
          )

        self.descale3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.descale4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.descale5 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

    def _make_multiple(self, nChannels, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(SingleLayer(nChannels))
        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        # input = input * 2.0 - 1.0

        scale1 = self.scale1(input)
        scale2 = self.scale2(scale1)
        scale3 = self.scale3(scale2)
        scale4 = self.scale3(scale3)
        scale5 = self.scale3(scale4)

        # print('sc1',scale1.shape)
        # print('sc2', scale2.shape)
        # print('sc3', scale3.shape)
        # print('sc4', scale4.shape)
        # print('sc5', scale5.shape)

        mid1 = self.multiple1(scale1)
        mid2 = self.multiple2(scale2)
        mid3 = self.multiple3(scale3)
        mid4 = self.multiple4(scale4)
        mid5 = self.multiple5(scale5)

        # print('mid1',mid1.shape)
        # print('mid2', mid2.shape)
        # print('mid3', mid3.shape)
        # print('mid4', mid4.shape)
        # print('mid5', mid5.shape)

        descale5 = self.descale5(mid5)
        descale4 = self.descale4(mid4)
        descale3 = self.descale3(mid3)
        descale2 = self.descale2(mid2)
        descale1 = self.descale1(mid1)

        # print('ds1',descale1.shape)
        # print('ds2', descale2.shape)
        # print('ds3', descale3.shape)
        # print('ds4', descale4.shape)
        # print('ds5', descale5.shape)

        output = descale1 + descale2 + descale3 + descale4 + descale5


        if self.output_all:
            return [descale1, descale2, descale3, descale4, descale5, output]

        else:
            return output