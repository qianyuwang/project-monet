import os
import torch
import numpy as np
import torchvision
import scipy.misc
import math
import torchvision.transforms as transforms

loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()

def threshold_func(image, block_size =3, delta= 5, max_value=1):
    '''
    :param image: type is torch.tensor, which is 4 dim, will be processed in this func
    :param block_size: must be odd number
    :param delta :   threshold=mean-delta
    :param max_value: 0 or 255
    :return: threshold now
    '''
    #print(dir(image))
    print('image',image.data.size())
    img = unloader(image.data.squeeze()) # previous error 'torch.tensor not tensor' is caused by did not squeeze
    img = img.convert('L')
  #  assert(block_size>1 && block_size%2==1)
    img = loader(img)
    size = img.size()  #torch.Size([1, 256, 256])
    #print(dir(img))
    new_img = torch.Tensor(size)


    print('threshold done')

    return 3
