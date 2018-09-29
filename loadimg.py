import torch.utils.data as data

from PIL import Image
import os
import os.path

def default_loader(path):
    img = Image.open(path)
    return img

def default_list_reader(file_dir):
    source = os.listdir(file_dir)
    imgList = []
    for dir in source:
        dir_tar = dir[:-10] + 'target.png'
        imgList.append((dir, dir_tar))
    return imgList

class ImageList(data.Dataset):
    def __init__(self, rootsour=None, roottar=None, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.rootsour      = rootsour
        self.roottar = roottar
        self.imgList   = list_reader(rootsour)
        self.transform = transform
        self.loader    = loader

    def __getitem__(self, index):
        imgsourName, imgtarName = self.imgList[index]
        imgsour = self.loader(os.path.join(self.rootsour,imgsourName))
        imgtar = self.loader(os.path.join(self.roottar, imgtarName))
        if self.transform is not None:
            imgsour = self.transform(imgsour)
            imgtar = self.transform(imgtar)
        return imgsour, imgtar

    def __len__(self):
        return len(self.imgList)