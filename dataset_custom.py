import struct
import pandas as pd
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

class CustomDataset(Dataset):
    def __init__(self, train_img_dir='', val_img_dir='', transform=None, target_transform=None,istrain=True):    
        self.train_img_dir = train_img_dir
        self.val_img_dir=val_img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels_tmp={'file_name':[],'label':[]}
        self.istrain= istrain
        if self.istrain:
            self.make_train_file()
            self.img_labels = pd.read_csv('train_file.csv', names=['file_name', 'label'])
        else:
            self.make_val_file()
            self.img_labels = pd.read_csv('val_file.csv', names=['file_name', 'label'])
    def make_train_file(self):
        if Path('train_file.csv').exists():
            return

        for name in tqdm(os.listdir(self.train_img_dir)):
            if name.split('.')[-1] in ['zip', 'irx']:
                continue
            tmp=name.split("_")
            tmp_label=tmp[0]
            f1=os.path.join(self.train_img_dir, name)
#            f2=os.path.join(self.annotation_dir,name)
            for img in os.listdir(f1):
                l=len(os.listdir(f1))-1
                isjpg=img.split('.')
                if isjpg[-1]=='jpg':
                    part=isjpg[0].split('_')
                    if part[-1]=='leaf':
                        pls=0
                    elif part[-1]=='flower':
                        pls=1
                    elif part[-1]=='stem':
                        pls=2
                    elif part[-1]=='fruit':
                        pls=3
                    elif part[-1]=='root':
                        pls=4
                    else:
                        continue
                    label=(int(tmp_label)-1)*5+pls
                    f3=os.path.join(f1,img)
                    if img=='desktop.ini':
                        continue
                    self.img_labels_tmp['file_name'].append(f3)
                    self.img_labels_tmp['label'].append(label)
        data=pd.DataFrame(data=self.img_labels_tmp)
        data.to_csv('train_file.csv', index=False, header=False)
    def make_val_file(self):
        if Path('val_file.csv').exists():
            return

        for name in os.listdir(self.val_img_dir):
            if name.split('.')[-1] in ['zip', 'irx']:
                continue
            tmp=name.split("_")
            tmp_label=tmp[0]
            f1=os.path.join(self.val_img_dir, name)
#            f2=os.path.join(self.annotation_dir,name)
            for i,img in enumerate(os.listdir(f1)):
                l=len(os.listdir(f1))-1
                isjpg=img.split('.')
                if isjpg[-1]=='jpg':
                    part=isjpg[0].split('_')
                    if part[-1]=='leaf':
                        pls=0
                    elif part[-1]=='flower':
                        pls=1
                    elif part[-1]=='stem':
                        pls=2
                    elif part[-1]=='fruit':
                        pls=3
                    elif part[-1]=='root':
                        pls=4
                    else:
                        continue
                label=(int(tmp_label)-1)*5+pls
                f3=os.path.join(f1,img)
                if img=='desktop.ini':
                    continue
                self.img_labels_tmp['file_name'].append(f3)
                self.img_labels_tmp['label'].append(label)
        data=pd.DataFrame(data=self.img_labels_tmp)
        data.to_csv('val_file.csv', index=False, header=False)
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        file = self.img_labels.iloc[idx, 0]
        img_path =os.path.join(os.path.abspath(os.getcwd()), self.img_labels.iloc[idx, 0])
        #print(img_path)
#        image = cv2.imread(img_path,cv2.IMREAD_COLOR)
        image = read_image(img_path)
        # image=np.swapaxes(image,0,2)
        # image=np.swapaxes(image,0,1)
        label = self.img_labels.iloc[idx, 1]
        image = self.transform(image/255)
        label=int(label)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__=='__main__':
    # image=read_image('./img/033_참당귀/033_00000003_leaf.jpg')
    # print(image)
    pass