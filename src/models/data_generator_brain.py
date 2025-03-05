# camera-ready

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms


import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import math
import scipy.stats

import pickle

import cv2

ROOT_DIR  = '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/data'

MAX_LABEL  = 362

class BrainDatasetTrain(torch.utils.data.Dataset):
    def __init__(self, model = None):
        with open(f"{ROOT_DIR}/BrainTumourPixels/labels_train.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open(f"{ROOT_DIR}/BrainTumourPixels/images_train.pkl", "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)

        self.num_examples = self.labels.shape[0]

        print ("DatasetTrain - number of images: %d" % self.num_examples)
        print (np.min(self.labels))
        print (np.max(self.labels))
        print (np.mean(self.labels))
        
        if model == 'densenet201':
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Resize to match DenseNet201 input
            transforms.ToTensor(),  # Converts to tensor & scales to [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
        ])
        elif model =='efficientnetb4':
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((380, 380)),  # Resize to EfficientNetB4's expected input size
            transforms.ToTensor(),  # Convert to tensor (scales pixel values to [0,1])
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1,1]
        ])
        else:
            self.transform = None

    def __getitem__(self, index):
        label = self.labels[index]
        img = self.imgs[index] # (shape: (64, 64, 3))

        if self.transform is None:
            img = img/255.0
            img = img - np.array([0.485, 0.456, 0.406])
            img = img/np.array([0.229, 0.224, 0.225])
            img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
            img = img.astype(np.float32)
        else:
            img = self.transform(img)  
        
        label = torch.tensor(label / MAX_LABEL, dtype=torch.float32) 

        return (img, label)

    def __len__(self):
        return self.num_examples

class BrainDatasetVal(torch.utils.data.Dataset):
    def __init__(self, model = None):
        with open(f"{ROOT_DIR}/BrainTumourPixels/labels_val.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open(f"{ROOT_DIR}/BrainTumourPixels/images_val.pkl", "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)

        self.num_examples = self.labels.shape[0]

        print ("DatasetVal - number of images: %d" % self.num_examples)
        print (np.min(self.labels))
        print (np.max(self.labels))
        print (np.mean(self.labels))
        
        if model == 'densenet201':
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Resize to match DenseNet201 input
            transforms.ToTensor(),  # Converts to tensor & scales to [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
        ])
        elif model =='efficientnetb4':
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((380, 380)),  # Resize to EfficientNetB4's expected input size
            transforms.ToTensor(),  # Convert to tensor (scales pixel values to [0,1])
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1,1]
        ])
        else:
            self.transform = None

    def __getitem__(self, index):
        label = self.labels[index]
        img = self.imgs[index] # (shape: (64, 64, 3))

        if self.transform is None:
            img = img/255.0
            img = img - np.array([0.485, 0.456, 0.406])
            img = img/np.array([0.229, 0.224, 0.225])
            img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
            img = img.astype(np.float32)
        else:
            img = self.transform(img)  
        
        label = torch.tensor(label / MAX_LABEL, dtype=torch.float32) 

        return (img, label)

    def __len__(self):
        return self.num_examples

class BrainDatasetTest(torch.utils.data.Dataset):
    def __init__(self, model = None):
        with open(f"{ROOT_DIR}/BrainTumourPixels/labels_test.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open(f"{ROOT_DIR}/BrainTumourPixels/images_test.pkl", "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)

        self.num_examples = self.labels.shape[0]

        print ("DatasetTest - number of images: %d" % self.num_examples)
        print (np.min(self.labels))
        print (np.max(self.labels))
        print (np.mean(self.labels))
        
        if model == 'densenet201':
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Resize to match DenseNet201 input
            transforms.ToTensor(),  # Converts to tensor & scales to [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
        ])
        elif model =='efficientnetb4':
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((380, 380)),  # Resize to EfficientNetB4's expected input size
            transforms.ToTensor(),  # Convert to tensor (scales pixel values to [0,1])
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1,1]
        ])
        else:
            self.transform = None

    def __getitem__(self, index):
        label = self.labels[index]
        img = self.imgs[index] # (shape: (64, 64, 3))

        if self.transform is None:
            img = img/255.0
            img = img - np.array([0.485, 0.456, 0.406])
            img = img/np.array([0.229, 0.224, 0.225])
            img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
            img = img.astype(np.float32)
        else:
            img = self.transform(img)  
        
        label = torch.tensor(label / MAX_LABEL, dtype=torch.float32) 

        return (img, label)

    def __len__(self):
        return self.num_examples