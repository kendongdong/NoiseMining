import os
import numpy as np
import torch
import PIL.Image as Image
from torch.utils import data


class MyData(data.Dataset):  # inherit
   
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])
    mean_focal = np.tile(mean_rgb, 12)
    std_focal = np.tile(std_rgb, 12)
    
    def __init__(self, params, data_root, data_list, noisy_root, transform=True):
        super(MyData, self).__init__()
        self.root = data_root
        self._transform = transform
        self.list_path = data_list
        self.noisy_path = noisy_root
        self.params = params
        self.list = None

        with open(self.list_path,'r') as file:
            self.list = [x.strip() for x in file.readlines()]
        file.close()

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        img_name = self.list[index] # same as lbl_name
        
        img = self.load_image(os.path.join(self.root, 'MSRA-B_image', img_name+'.jpg'))
        lbl = self.load_sal_label(os.path.join(self.root, 'MSRA-B_mask',img_name+'.png'))
    
        noisy_lbl = []
        for noise_path in self.noisy_path:
            noisy_lbl.append(torch.Tensor(self.load_noisy_label(os.path.join(noise_path, img_name+'.png'))))
            
        if self._transform:
            img = self.transform(img)
             
        img = torch.Tensor(img)
        lbl = torch.Tensor(lbl)
        noisy_lbl = torch.stack(noisy_lbl)
        
        
        sample = {'image':img, 'label':lbl, 'noisy_label':noisy_lbl,'img_name':img_name,'idx': index}
        return sample
        
    # Translating numpy_array into format that pytorch can use on Code.
    def transform(self, img):

        img = img.astype(np.float64)/255.0
        img -= self.mean_rgb
        img /= self.std_rgb
        img = img.transpose(2, 0, 1)  # to verify
        img = torch.from_numpy(img).float()
        return img

    # load image
    def load_image(self, path, noise=False):
        if not os.path.exists(path):
            print('File {} not exists'.format(path))
        img = Image.open(path)
        img = img.resize((256,256))
        img = np.array(img, dtype=np.int32)
        return img

    # load noisy label
    def load_sal_label(self, path):
        if not os.path.exists(path):
            print('File {} not exists'.format(path))
        im = Image.open(path)
        im = im.resize((256,256))
        label = np.array(im, dtype=np.int32)
        
        if len(label.shape) == 3:
            label = label[:,:,0]
            label[label!=0] = 1
            label = label[np.newaxis, ...]
        else:
            label[label!=0] = 1
            label = label[np.newaxis, ...]
        return label

    def load_noisy_label(self, path):
        if not os.path.exists(path):
            print('File {} not exists'.format(path))
        im = Image.open(path)
        im = im.resize((256,256))
        label = np.array(im, dtype=np.int32)
     
        if len(label.shape) == 3:
            label = label[:,:,0]
        label = label / 255.
        label = label * 10
        label = label.astype(np.int)
        label = label.astype(np.float)
        return label

class MyTestData(data.Dataset):

    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])
    mean_focal = np.tile(mean_rgb, 12)
    std_focal = np.tile(std_rgb, 12)

    def __init__(self, params, data_root, data_list, transform=True):
        super(MyTestData, self).__init__()
        self.root = data_root
        self.list_path = data_list
        self._transform = transform

        with open(self.list_path, 'r') as file:
            self.list = [x.strip() for x in file.readlines()]
        file.close()

        self.test_num = len(self.list)

    def __len__(self):
        return self.test_num

    def __getitem__(self, index):
        img_name = self.list[index % self.test_num]  # same as lbl_name
      
        img = self.load_image(os.path.join(self.root, 'MSRA-B_image', img_name + '.jpg'))
        lbl = self.load_sal_label(os.path.join(self.root, 'MSRA-B_mask', img_name + '.png'))
       
        if self._transform:
            img,focal = self.transform(img,focal)
            
        img = torch.Tensor(img)
        lbl = torch.Tensor(lbl)
        
        sample = {'image':img, 'label':lbl, 'img_name':img_name,'idx': index,}
        return sample

    def transform(self, img):
        img = img.astype(np.float64)/255.0
        img -= self.mean_rgb
        img /= self.std_rgb
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img

    # load image
    def load_image(self, path, noise=False):
        if not os.path.exists(path):
            print('File {} not exists'.format(path))
        img = Image.open(path)
        img = img.resize((256,256))
        img = np.array(img, dtype=np.int32)
        return img

    # load noisy label
    def load_sal_label(self, path):
        if not os.path.exists(path):
            print('File {} not exists'.format(path))
        im = Image.open(path)
        im = im.resize((256,256))
        label = np.array(im, dtype=np.int32)
        if len(label.shape) == 3:
            label = label[:,:,0]
            label = label / 255.
            label = label[np.newaxis, ...]
            label[label!=0] = 1
        return label