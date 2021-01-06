from collections import defaultdict
import copy
import random
import numpy as np
import os
import shutil
from urllib.request import urlretrieve
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from utils import Mixup, RandAugment, AsymmetricLossSingleLabel, SCELoss, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from PIL import Image

def merge_data(df1, df2):
    merge_df = pd.concat([df1, df2], axis=0) #,how ='outer', on ='image_id')
    return merge_df

def balance_data(df, mode="undersampling", val=False):
    class_0 = df[df.label==0]
    class_1 = df[df.label==1]
    class_2 = df[df.label==2]
    class_3 = df[df.label==3]
    class_4 = df[df.label==4]
    if mode == "undersampling":
        # upsample minority
        class_3_downsampled = resample(class_3,
                                  replace=True, # sample with replacement
                                  n_samples=int(len(class_3)*2/3), # match number in majority class
                                  random_state=27) # reproducible results
        if val:
            return  pd.concat([class_0, class_1, class_2, class_3_downsampled, class_4]) 
    
        class_1_downsampled = resample(class_1,
                          replace=True, # sample with replacement
                          n_samples=int(len(class_1)*0.7), # match number in majority class
                          random_state=27) # reproducible results
        class_4_upsampled = resample(class_4,
                          replace=True, # sample with replacement
                          n_samples=int(len(class_4)*1.3), # match number in majority class
                          random_state=27) # reproducible results
        return pd.concat([class_0, class_1_downsampled, class_2, class_3_downsampled, class_4_upsampled]) 
    else:
        class_0_upsampled = resample(class_0,
                          replace=True, # sample with replacement
                          n_samples=int(len(class_3)/4), # match number in majority class
                          random_state=27) # reproducible results
        class_1_upsampled = resample(class_1,
                          replace=True, # sample with replacement
                          n_samples=int(len(class_3)/3), # match number in majority class
                          random_state=27) # reproducible results
        class_2_upsampled = resample(class_2,
                          replace=True, # sample with replacement
                          n_samples=int(len(class_3)/3), # match number in majority class
                          random_state=27) # reproducible results
        class_4_upsampled = resample(class_4,
                          replace=True, # sample with replacement
                          n_samples=int(len(class_3)/3), # match number in majority class
                          random_state=27) # reproducible results
        return pd.concat([class_0_upsampled, class_1_upsampled, class_2_upsampled, class_3, class_4_upsampled]) 


# Dataset
class TrainDataset(Dataset):
    def __init__(self, df, root, transform=None, mosaic_mix = False, soft_df = None):
        self.df = df
        self.file_names = df['image_id'].values
        self.labels = df['label'].values
        self.transform = transform
        self.mosaic_mix = mosaic_mix
        self.rand_aug_fn = None #RandAugment()
        self.distill_soft_target = soft_df 
        self.root = root
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{self.root}/train_images/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = torch.tensor(self.labels[idx]).long()
        if self.rand_aug_fn is not None:
            image = np.array(self.rand_aug_fn(Image.fromarray(image)))
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        if self.distill_soft_target is not None:
            try:
                soft_label = [float(t) for t in soft_target_distill[idx][0].split(" ")]
                soft_label = torch.tensor(soft_label)
            except:
                soft_label = torch.tensor([0., 0., 0., 0., 0.])
        else:
            soft_label = 0.
        return image, label, soft_label, file_name
    

class TestDataset(Dataset):
    def __init__(self, df, root, transform=None, valid_test=False, fcrops=False):
        self.df = df
        self.root = root
        self.file_names = df['image_id'].values
        self.transform = transform
        self.valid_test = valid_test
        self.fcrops = fcrops
        if self.valid_test:
            self.labels = df['label'].values  
        else:
            assert ValueError("Test data does not have annotation, plz check!")
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        if self.valid_test:
            file_path = f'{self.root}/train_images/{file_name}'
            #file_path = f'{self.root}/external/extraimages/{file_name}'
        else:
            file_path = f'{self.root}/test_images/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if isinstance(self.transform, list):
            outputs = {'images':[],
                       'labels':[],
                       'image_ids':[]}
            if self.fcrops:
                for trans in self.transform:
                    image_aug = transforms.ToPILImage()(image)
                    image_aug = trans(image_aug)
                    outputs["images"].append(image_aug)
                    del image_aug
            else:
                for trans in self.transform:
                    augmented = trans(image=image)
                    image_aug = augmented['image']
                    outputs["images"].append(image_aug)
                    del image_aug

            if self.valid_test:
                label = torch.tensor(self.labels[idx]).long()
                outputs['labels'] = len(self.transform)*[label]
                outputs['image_ids'].append(file_name)
                
            else:
                outputs['labels'] = len(self.transform)*[-1]
                
            return outputs
        else:
            augmented = self.transform(image=image)
            image = augmented['image'] 
        return image
