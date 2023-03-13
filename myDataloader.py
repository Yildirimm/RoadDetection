from __future__ import print_function, division
import os

import cv2
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import segmentation_models_pytorch as smp
from PIL import Image

import my_unet as mu

### DATASET LOADER

class myDataset(Dataset):

    def __init__(self, images_dir, targets_dir=None, class_rgb_values=None, augmentation=None, preprocessing=None,
                 transform=None, indices=None):

        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.target_paths = [os.path.join(targets_dir, image_id) for image_id in
                             sorted(os.listdir(targets_dir))]
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.transform = transform

    def __len__(self):
        # return length of
        print(f'{self.image_paths[-1]} ->', len(self.image_paths))
        return len(self.image_paths)

    def __getitem__(self, idx):
        # image = io.imread(fname=self.image_paths[idx])
        # mask = io.imread(fname=self.target_paths[idx])
        # print(self.image_paths[idx], self.target_paths[idx])
        # print('Image and mask are loaded')

        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.target_paths[idx]), cv2.COLOR_BGR2RGB)

        print(idx)
        print('cv2 is successful')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # https://blog.csdn.net/weixin_41995638/article/details/120796353
        if self.transform is not None:
            transformed_img = self.transform(image=image)
            image = transformed_img["image"]

            transformed_mask = self.transform(image=mask)
            mask = transformed_mask["image"]

        print(f'image and mask type: ', type(image), type(mask))
        print(f'image and mask type: ', type(image[0, 0, 0]), '\n', image[0, 0, 0])

        return image, mask