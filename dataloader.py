import cv2
import numpy as np
import torch
import torchvision
import opencv_transforms.functional as FF
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os
from torchvision import datasets
from PIL import Image
    
class GetImageFolder(datasets.ImageFolder):   
    def __init__(self, root, transform, is_pair=False, refer_transform=None, start='gray', end='color', sketch_net=None):
        super(GetImageFolder, self).__init__(root, transform)
        self.sketch_net = sketch_net
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_pair = is_pair
        self.refer_transform = refer_transform
        self.start = start
        self.end = end
        
    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.loader(path)
        img = np.asarray(img)
        
        if self.is_pair:
            img = img[:, 0:512, :]
            
        img = self.transform(img)
        img_color = img
        
        if self.start == 'gray' or self.end == 'gray':
            #img_gray = FF.to_grayscale(img, num_output_channels=3)
            img_gray = convert_color(img, 'to_lab')[:, :, 0:1]
        
        if self.start == 'edge' or self.end == 'edge':
            with torch.no_grad():
                img_temp = make_tensor(img_color)
                img_edge = self.sketch_net(img_temp.unsqueeze(0).to(self.device)).squeeze().permute(1,2,0).cpu().numpy()
                img_edge = FF.to_grayscale(img_edge, num_output_channels=1)
                
        if self.start == 'color':
            img_start = img_color
        elif self.start == 'gray':
            img_start = img_gray
        elif self.start == 'edge':
            img_start = img_edge
        elif self.start == 'ab':
            img_start = img_color[:,:,1:3]
        elif self.start == 'left_half':
            img_start = img_color[:, 0:512, :]
        elif self.start == 'right_half':
            img_start = img_color[:, 512:1024, :]
        img_start = make_tensor(img_start)
            
        if self.end == 'color':
            img_end = img_color
        elif self.end == 'gray':
            img_end = img_gray
        elif self.end == 'edge':
            img_end = img_edge    
        elif self.end == 'ab':
            img_end = img_color[:,:,1:3]
        elif self.end == 'left_half':
            img_start = img_color[:, 0:512, :]
        elif self.end == 'right_half':
            img_start = img_color[:, 512:1024, :]
        img_end = make_tensor(img_end)
            
        img_refer = img_color
        if self.refer_transform:
            img_refer = self.refer_transform(img)
            img_refer = make_tensor(img_refer)

        return img_start, img_end, img_refer
    
class PairImageFolder():   
    def __init__(self, root, transform, mode):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dirA = root+'/'+mode+'A/'
        self.dirB = root+'/'+mode+'B/'
        self.imgsA = os.listdir(self.dirA)
        self.imgsB = os.listdir(self.dirB)
        self.transform = transform
        
    def __getitem__(self, index):
        imgA = cv2.imread(self.dirA + self.imgsA[index])
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)   
        imgA = self.transform(imgA)
        imgA = make_tensor(imgA)
        
        imgB = cv2.imread(self.dirB + self.imgsB[index])
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)   
        imgB = self.transform(imgB)
        imgB = make_tensor(imgB)
        
        return imgA, imgB
    
    def __len__(self):
        return min(len(self.imgsA), len(self.imgsB))
    
def make_tensor(img):
    img = FF.to_tensor(img)
    img = FF.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return img
    
def show_example(tensor_list, size):
    n = len(tensor_list)
    plt.figure(figsize=size)
    plt.subplots_adjust(hspace=0, wspace=0)

    for i in range(1, n+1):
        ax1 = plt.subplot(1, n, i)
        result =torch.cat([tensor_list[i-1]],dim=-1)
        plt.imshow(np.transpose(vutils.make_grid(result, nrow=1, padding=5, normalize=True).cpu(),(1,2,0)), aspect='auto')
        plt.axis("off")
    
    plt.show()
    
def convert_color(img, mode):
    b_size = 1
    
    if torch.is_tensor(img):
        img = img.squeeze().permute(1,2,0).cpu().numpy()

    if mode=='to_rgb':
        temp_img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    elif mode=='to_lab':
        temp_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)   

    return img

def concat_lab(img_l, img_ab):
    if len(img_l.size()) == 3:
        img_l = img_l.unsqueeze(1)
    img_lab = torch.cat([img_l, img_ab], dim=1)
    return img_lab

def invertColor(img):
    return 255 - img

def colorDodge(base, mix):
    base_i32 = base.astype(np.int32)
    mix_i32 = mix.astype(np.int32)
    divisor = 255 - mix
    posto255 = divisor == 0
    divisor[posto255] = 1
    ret = base_i32 + (base_i32 * mix_i32) / divisor
    ret[posto255] = 255
    ret[ret > 255] = 255
    return ret.astype(np.uint8)

def sobel(img):

    img_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=1, scale=1.5, delta=0, borderType=cv2.BORDER_DEFAULT)
    img_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=1, scale=1.5, delta=0, borderType=cv2.BORDER_DEFAULT)

    abs_img_x = cv2.convertScaleAbs(img_x)
    abs_img_y = cv2.convertScaleAbs(img_y)

    res = cv2.addWeighted(abs_img_x, 0.5, abs_img_y, 0.5, 0)
    return invertColor(res)

def threshold(img, threshold = 240):
    img[img>threshold] = 255

def enhance(img, threshold = 240, alpha = 0.8):
    pos = img <= threshold
    img = img.astype(np.float32)
    img[pos] *= alpha
    return img.astype(np.uint8)

def XDoG(img, sigma = 0.7, k = 3.0, t = 0.998, e = -0.1, p = 30):
    img = img.astype(np.float32)/255

    Ig1 = cv2.GaussianBlur(img, (3, 3), sigma, sigma)
    Ig2 = cv2.GaussianBlur(img, (3, 3), sigma * k, sigma * k)

    Dg = (Ig1 - t * Ig2)

    Dg[Dg<e] = 1
    Dg[Dg>=e]= 1 + np.tanh(p * Dg[Dg>=e])

    Dg[Dg>1.0] = 1.0
    Dg = Dg * 255

    return Dg.astype(np.uint8)

def sketch(img, mode = 'XDoG'):
    if mode == 'sobel':
        s = sobel(img)
        threshold(s)
        return s
    elif mode == 'erode':
        ivt = invertColor(img)
        mix = cv2.erode(ivt, (3,3), iterations=2)
        # mix = cv2.GaussianBlur(ivt, (3, 3), 2, 2)
        cd = colorDodge(img, mix)
        threshold(cd)
        return enhance(cd)
    elif mode == 'XDoG':
        return XDoG(img)