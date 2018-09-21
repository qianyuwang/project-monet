import torch
import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def readlinesFromFile(path, datasize):
    print("Load from file %s" % path)
    f=open(path)
    data=[]    
    for idx in xrange(0, datasize):
      line = f.readline()
      data.append(line)      
    
    f.close()  
    return data  


def loadFromFile(path, datasize):
    if path is None:
      return None, None
      
    print("Load from file %s" % path)
    f=open(path)
    data=[]
    label=[]
    for idx in xrange(0, datasize):
      line = f.readline().split()
      data.append(line[0])         
      label.append(line[1])
       
    f.close()  
    return data, label     


def load_video_image(file_path, input_height=None, input_width=None, output_height=None, output_width=None,
              crop_height=None, crop_width=None, is_random_crop=True, is_mirror=True,
              is_gray=True):
    
    if input_width is None:
        input_width = input_height
    if output_width is None:
        output_width = output_height
    if crop_width is None:
        crop_width = crop_height
    
    img = Image.open(file_path)

    if is_gray is False and img.mode is not 'RGB':
        img = img.convert('RGB')
    if is_gray and img.mode is not 'L':
        img = img.convert('L')

    [w, h] = img.size
    img_mo = ImageOps.crop(img, (0, 0, w//2, 0))
    img_cl = ImageOps.crop(img, (w//2, 0, 0, 0))

    if is_mirror and random.randint(0,1) is 0:
        img_mo = ImageOps.mirror(img_mo)
        img_cl = ImageOps.mirror(img_cl)
      
    if input_height is not None:
        img_mo = img_mo.resize((input_width, input_height),Image.BICUBIC)
        img_cl = img_cl.resize((input_width, input_height), Image.BICUBIC)

    if crop_height is not None:
        if is_random_crop:
            cx1 = random.randint(0, w-crop_width)
            cx2 = w - crop_width - cx1
            cy1 = random.randint(0, h-crop_height)
            cy2 = h - crop_height - cy1
        else:
            cx2 = cx1 = int(round((w-crop_width)/2.))
            cy2 = cy1 = int(round((h-crop_height)/2.))
        img_mo = ImageOps.crop(img_mo, (cx1, cy1, cx2, cy2))
        img_cl = ImageOps.crop(img_cl, (cx1, cy1, cx2, cy2))

    img_mo = img_mo.resize((int(output_width), int(output_height)),Image.BICUBIC)
    img_cl = img_cl.resize((int(output_width), int(output_height)),Image.BICUBIC)

    return img_mo, img_cl
      
      
class ImageDatasetFromFile(data.Dataset):
    def __init__(self, root_path,  input_height=None, input_width=None, output_height=64,output_width=64, crop_height=288, crop_width=288,
                 is_random_crop=True, is_mirror=True, is_gray=False):
        super(ImageDatasetFromFile, self).__init__()
                
        self.image_filenames = [join(root_path, x) for x in listdir(root_path) if is_image_file(x)]
        self.is_random_crop = is_random_crop
        self.is_mirror = is_mirror
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.root_path = root_path
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.is_gray = is_gray
                       
        self.input_transform = transforms.Compose([ 
                                   transforms.ToTensor()                                                                      
                               ])

    def __getitem__(self, index):
    
        if self.is_mirror:
            is_mirror = random.randint(0,1) is 0
        else:
            is_mirror = False
          
        mo_img, cl_img = load_video_image(self.image_filenames[index],
                                  self.input_height, self.input_width, self.output_height, self.output_width,
                                  self.crop_height, self.crop_width, self.is_random_crop, is_mirror,
                                  self.is_gray)
        
        
        input = self.input_transform(mo_img)
        target = self.input_transform(cl_img)
        
        return input, target

    def __len__(self):
        return len(self.image_filenames)