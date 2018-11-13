import torch.utils.data as data

from PIL import Image
import os
import os.path

def default_loader(path):
    img = Image.open(path)
    return img

def loader_L(path):
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

def real_data_list_reader(file_dir):
    source = os.listdir(file_dir)
    source.sort()
    imgList = []
    for dir in source:
        imgList.append((dir))
    return imgList

class ImageList(data.Dataset):
    def __init__(self, rootsour=None, roottar=None,rootgt=None, transform=None, real_data= False):
        self.rootsour = rootsour
        self.roottar  = roottar
        self.rootgt   =rootgt
        self.transform = transform
        self.loader    = default_loader
        self.loader_L = loader_L
        self.real_data = real_data
        if self.real_data:
            self.list_reader = real_data_list_reader
        else:
            self.list_reader = default_list_reader
        self.imgList = self.list_reader(rootsour)
    def __getitem__(self, index):
        if self.real_data:
            imgsourName = self.imgList[index]
            imgsour = self.loader_L(os.path.join(self.rootsour,imgsourName))
            if self.transform is not None:
                imgsour = self.transform(imgsour)
            return imgsour
        else:
            imgsourName, imgtarName,imggtName = self.imgList[index]
            imgsour = self.loader_L(os.path.join(self.rootsour,imgsourName))
            imgtar = self.loader_L(os.path.join(self.roottar, imgtarName))
            imggt = self.loader_L(os.path.join(self.rootgt, imggtName))
            if self.transform is not None:
                imgsour = self.transform(imgsour)
                imgtar = self.transform(imgtar)
                imggt  =self.transform(imggt)
            return imgsour, imgtar,imggt

    def __len__(self):
        return len(self.imgList)
