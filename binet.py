import torch
import torch.nn as nn
import math

class BiCNN(nn.Module):
    def __init__(self, output_all=False, model_path=None):
        super(BiCNN, self).__init__()

        self.threshold = nn.Conv2d(1, 1, 3, 1, 1)
        self.lamda = 10000
        self.binary = nn.Sigmoid()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         #nn.init.constant(m.weight, 0.3)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    # def _initialize_weights(self):
    #     print('goin init')
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             m.weight.data.normal_(0, 0.01)
    #             m.bias.data.zero_()



    def forward(self, x):
        output = x

        output_gray = 0.299 * output[:, 0, :, :] + 0.587 * output[:, 1, :, :] + 0.114 * output[:, 2, :, :]
        output_gray = output_gray.unsqueeze(0)

        thresold_gray = self.threshold(output_gray)
        output_bina = output_gray - thresold_gray
        output_bina = self.binary(output_bina)

        return output_bina
