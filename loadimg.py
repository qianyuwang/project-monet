import torch.utils.data as data

from PIL import Image
import os
import os.path

def default_loader(path):
    img = Image.open(path)
    return img
def default_loader_L(path):
    img = Image.open(path)
    img_L = img.convert('L')
    return img_L
def default_list_reader(file_dir):
    source = os.listdir(file_dir)
    source.sort()  
    imgList = []
    for dir in source:
        dir_tar = dir.split('.',1)[0]+'.png'
        dir_gt= dir.split('.',1)[0]+'.png'
        imgList.append((dir, dir_tar,dir_gt))#在列表末尾添加元素
    return imgList

class ImageList(data.Dataset):
    def __init__(self, rootsour=None, roottar=None,rootgt=None, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.rootsour = rootsour
        self.roottar  = roottar
        self.rootgt   =rootgt
        self.imgList   = list_reader(rootsour)
        self.transform = transform
        self.loader    = loader
        self.loader_L = default_loader_L

    def __getitem__(self, index):
        imgsourName, imgtarName,imggtName = self.imgList[index]
        imgsour = self.loader(os.path.join(self.rootsour,imgsourName))
        imgtar = self.loader(os.path.join(self.roottar, imgtarName))
        imggt = self.loader_L(os.path.join(self.rootgt, imggtName))
        if self.transform is not None:
            imgsour = self.transform(imgsour)
            imgtar = self.transform(imgtar)
            imggt  =self.transform(imggt)
        return imgsour, imgtar,imggt

    def __len__(self):
        return len(self.imgList)
