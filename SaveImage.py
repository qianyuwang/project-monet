import os
import torch
import numpy as np
import torchvision
import scipy.misc
import math


def save_images(epoch,images, name, nrow=10):
  image_save_path = "results/epoch_{}/".format(epoch)
  if not os.path.exists("results/epoch_{}/".format(epoch)):
    os.makedirs("results/epoch_{}/".format(epoch))   
  #print(images.size())
  img = images.cpu()
  im = img.data.numpy().astype(np.float32)
     
  im = im.transpose(0,2,3,1)
  imsave(im, [nrow, int(math.ceil(float(im.shape[0])/nrow))], "results/epoch_{}/".format(epoch)+name) 

def merge(images, size):
  #print(images.shape())
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3)) 
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  return img

def imsave(images, size, path):
  return scipy.misc.imsave(path, merge(images, size))  
