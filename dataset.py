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
              is_random_crop=True, is_mirror=True,
              is_gray=False, is_crop= True , is_duplicate = False, is_train =False):
    
    if input_width is None:
        input_width = input_height
    if output_width is None:
        output_width = output_height
    input_transform = transforms.Compose([
        transforms.ToTensor()
    ])
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

    if is_crop:
        w = w//2
        if is_duplicate:
            crop_width = w - w % 128
            crop_height = h - h % 128
        else:
            crop_width = w - w % 256
            crop_height = h - h % 256
        if is_random_crop and is_train:
            print('into crop')
            cx1 = random.randint(0, w-crop_width)
            cx2 = w - crop_width - cx1
            cy1 = random.randint(0, h-crop_height)
            cy2 = h - crop_height - cy1
        else:
            cx2 = cx1 = int(round((w-crop_width)/2.))
            cy2 = cy1 = int(round((h-crop_height)/2.))
        img_mo = ImageOps.crop(img_mo, (cx1, cy1, cx2, cy2))
        img_cl = ImageOps.crop(img_cl, (cx1, cy1, cx2, cy2))
    print('into 1')
   # if is_train:
    print('into2')
    [w, h] = img_mo.size
    img_mos= {}
    img_cls = {}
    if is_duplicate:
        pix = 128
        pp = 256
        img_mos['1_1'] = img_mo1_1 = ImageOps.crop(img_mo, (0,     0,    w - pp,           h - pp))
        img_mos['2_1'] = img_mo2_1 = ImageOps.crop(img_mo, (pix,   0,    w - pp - pix,     h - pp))
        print("img_mos['2_1']",img_mos['2_1'])
        img_mo3_1 = ImageOps.crop(img_mo, (2*pix, 0,    w - pp - 2*pix,   h - pp))
        img_mo4_1 = ImageOps.crop(img_mo, (3*pix, 0,    w - pp - 3*pix,   h - pp))
        img_mo5_1 = ImageOps.crop(img_mo, (4*pix, 0,    w - pp - 4*pix,   h - pp))
        img_mo6_1 = ImageOps.crop(img_mo, (5*pix, 0,    w - pp - 5*pix,   h - pp))
        img_mo1_2 = ImageOps.crop(img_mo, (0,     pix,    w - pp,           h - pp- pix))
        img_mo2_2 = ImageOps.crop(img_mo, (pix,   pix,    w - pp - pix,     h - pp- pix))
        img_mo3_2 = ImageOps.crop(img_mo, (2*pix, pix,    w - pp - 2*pix,   h - pp- pix))
        img_mo4_2 = ImageOps.crop(img_mo, (3*pix, pix,    w - pp - 3*pix,   h - pp- pix))
        img_mo5_2 = ImageOps.crop(img_mo, (4*pix, pix,    w - pp - 4*pix,   h - pp- pix))
        img_mo6_2 = ImageOps.crop(img_mo, (5*pix, pix,    w - pp - 5*pix,   h - pp- pix))
        img_mo1_3 = ImageOps.crop(img_mo, (0,     2*pix,    w - pp,           h - pp- 2*pix))
        img_mo2_3 = ImageOps.crop(img_mo, (pix,   2*pix,    w - pp - pix,     h - pp- 2*pix))
        img_mo3_3 = ImageOps.crop(img_mo, (2*pix, 2*pix,    w - pp - 2*pix,   h - pp- 2*pix))
        img_mo4_3 = ImageOps.crop(img_mo, (3*pix, 2*pix,    w - pp - 3*pix,   h - pp- 2*pix))
        img_mo5_3 = ImageOps.crop(img_mo, (4*pix, 2*pix,    w - pp - 4*pix,   h - pp- 2*pix))
        img_mo6_3 = ImageOps.crop(img_mo, (5*pix, 2*pix,    w - pp - 5*pix,   h - pp- 2*pix))
        img_mo1_4 = ImageOps.crop(img_mo, (0,     3*pix,    w - pp,           h - pp- 3*pix))
        img_mo2_4 = ImageOps.crop(img_mo, (pix,   3*pix,    w - pp - pix,     h - pp- 3*pix))
        img_mo3_4 = ImageOps.crop(img_mo, (2*pix, 3*pix,    w - pp - 2*pix,   h - pp- 3*pix))
        img_mo4_4 = ImageOps.crop(img_mo, (3*pix, 3*pix,    w - pp - 3*pix,   h - pp- 3*pix))
        img_mo5_4 = ImageOps.crop(img_mo, (4*pix, 3*pix,    w - pp - 4*pix,   h - pp- 3*pix))
        img_mo6_4 = ImageOps.crop(img_mo, (5*pix, 3*pix,    w - pp - 5*pix,   h - pp- 3*pix))
        img_mo1_5 = ImageOps.crop(img_mo, (0,     4*pix,    w - pp,           h - pp- 4*pix))
        img_mo2_5 = ImageOps.crop(img_mo, (pix,   4*pix,    w - pp - pix,     h - pp- 4*pix))
        img_mo3_5 = ImageOps.crop(img_mo, (2*pix, 4*pix,    w - pp - 2*pix,   h - pp- 4*pix))
        img_mo4_5 = ImageOps.crop(img_mo, (3*pix, 4*pix,    w - pp - 3*pix,   h - pp- 4*pix))
        img_mo5_5 = ImageOps.crop(img_mo, (4*pix, 4*pix,    w - pp - 4*pix,   h - pp- 4*pix))
        img_mo6_5 = ImageOps.crop(img_mo, (5*pix, 4*pix,    w - pp - 5*pix,   h - pp- 4*pix))
        img_mo1_6 = ImageOps.crop(img_mo, (0,     5*pix,    w - pp,           h - pp- 5*pix))
        img_mo2_6 = ImageOps.crop(img_mo, (pix,   5*pix,    w - pp - pix,     h - pp- 5*pix))
        img_mo3_6 = ImageOps.crop(img_mo, (2*pix, 5*pix,    w - pp - 2*pix,   h - pp- 5*pix))
        img_mo4_6 = ImageOps.crop(img_mo, (3*pix, 5*pix,    w - pp - 3*pix,   h - pp- 5*pix))
        img_mo5_6 = ImageOps.crop(img_mo, (4*pix, 5*pix,    w - pp - 4*pix,   h - pp- 5*pix))
        img_mo6_6 = ImageOps.crop(img_mo, (5*pix, 5*pix,    w - pp - 5*pix,   h - pp- 5*pix))

        img_mos = zip(img_mo1_1,img_mo1_2,img_mo1_3,img_mo2_1,img_mo2_2,img_mo2_3,img_mo3_1,img_mo3_2,img_mo3_3)
        img_cls = zip(img_cl1_1, img_cl1_2, img_cl1_3, img_cl2_1, img_cl2_2, img_cl2_3, img_cl3_1, img_cl3_2, img_cl3_3)

    else:
        print('into not du')
        pix = 256
        pp = 256
        img_mo1_1 = ImageOps.crop(img_mo, (0,     0,    w - pp,           h - pp))
        img_mo2_1 = ImageOps.crop(img_mo, (pix,   0,    w - pp - pix,     h - pp))
        img_mo3_1 = ImageOps.crop(img_mo, (2*pix, 0,    w - pp - 2*pix,   h - pp))
        img_mo1_2 = ImageOps.crop(img_mo, (0,     pix,    w - pp,           h - pp- pix))
        img_mo2_2 = ImageOps.crop(img_mo, (pix,   pix,    w - pp - pix,     h - pp- pix))
        img_mo3_2 = ImageOps.crop(img_mo, (2*pix, pix,    w - pp - 2*pix,   h - pp- pix))
        img_mo1_3 = ImageOps.crop(img_mo, (0,     2*pix,    w - pp,           h - pp- 2*pix))
        img_mo2_3 = ImageOps.crop(img_mo, (pix,   2*pix,    w - pp - pix,     h - pp- 2*pix))
        img_mo3_3 = ImageOps.crop(img_mo, (2*pix, 2*pix,    w - pp - 2*pix,   h - pp- 2*pix))
        img_cl1_1 = ImageOps.crop(img_cl, (0,     0,    w - pp,           h - pp))
        img_cl2_1 = ImageOps.crop(img_cl, (pix,   0,    w - pp - pix,     h - pp))
        img_cl3_1 = ImageOps.crop(img_cl, (2*pix, 0,    w - pp - 2*pix,   h - pp))
        img_cl1_2 = ImageOps.crop(img_cl, (0,     pix,    w - pp,           h - pp- pix))
        img_cl2_2 = ImageOps.crop(img_cl, (pix,   pix,    w - pp - pix,     h - pp- pix))
        img_cl3_2 = ImageOps.crop(img_cl, (2*pix, pix,    w - pp - 2*pix,   h - pp- pix))
        img_cl1_3 = ImageOps.crop(img_cl, (0,     2*pix,    w - pp,           h - pp- 2*pix))
        img_cl2_3 = ImageOps.crop(img_cl, (pix,   2*pix,    w - pp - pix,     h - pp- 2*pix))
        img_cl3_3 = ImageOps.crop(img_cl, (2*pix, 2*pix,    w - pp - 2*pix,   h - pp- 2*pix))

        img_mos['1_1'] = input_transform(img_mo1_1)
        img_mos['2_1'] = input_transform(img_mo2_1)
        img_mos['3_1'] = input_transform(img_mo3_1)
        img_mos['1_2'] = input_transform(img_mo1_2)
        img_mos['2_2'] = input_transform(img_mo2_2)
        img_mos['3_2'] = input_transform(img_mo3_2)
        img_mos['1_3'] = input_transform(img_mo1_3 )
        img_mos['2_3'] = input_transform(img_mo2_3 )
        img_mos['3_3'] = input_transform(img_mo3_3 )
        img_cls['1_1'] = input_transform(img_cl1_1)
        img_cls['2_1'] = input_transform(img_cl2_1)
        img_cls['3_1'] = input_transform(img_cl3_1)
        img_cls['1_2'] = input_transform(img_cl1_2)
        img_cls['2_2'] = input_transform(img_cl2_2)
        img_cls['3_2'] = input_transform(img_cl3_2)
        img_cls['1_3'] = input_transform(img_cl1_3)
        img_cls['2_3'] = input_transform(img_cl2_3)
        img_cls['3_3'] = input_transform(img_cl3_3)

    return img_mos, img_cls

    # else:
    #     print('into test')
    #     img_mo = input_transform(img_mo)
    #     img_cl = input_transform(img_cl)
    #     return img_mo,img_cl



      
      
class ImageDatasetFromFile(data.Dataset):
    def __init__(self, root_path,  input_height=None, input_width=None, output_height=256,output_width=256,
                 is_random_crop=True, is_mirror=True, is_gray=False , is_duplicate = False , is_crop =True,
                 is_train=False):
        super(ImageDatasetFromFile, self).__init__()
                
        self.image_filenames = [join(root_path, x) for x in listdir(root_path) if is_image_file(x)]
        self.is_random_crop = is_random_crop
        self.is_mirror = is_mirror
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.root_path = root_path
        self.is_gray = is_gray
        self.is_crop = is_crop
        self.is_duplicate  = is_duplicate
        self.is_train =is_train

    def __getitem__(self, index):
    
        if self.is_mirror:
            is_mirror = random.randint(0,1) is 0
        else:
            is_mirror = False
          
        mo_img, cl_img = load_video_image(self.image_filenames[index],
                                  self.input_height, self.input_width, self.output_height, self.output_width,
                                  self.is_random_crop, is_mirror,
                                  self.is_gray,self.is_crop,self.is_duplicate,self.is_train)
        

        return mo_img, cl_img

    def __len__(self):
        return len(self.image_filenames)