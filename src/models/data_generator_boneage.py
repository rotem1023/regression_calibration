import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os
import zipfile
import threading



# class BoneAgeDataset(Dataset):
#     """
#     Loads the rsna bone age data set
#     """

#     def __init__(self, data_dir='/home/dsi/rotemnizhar/.cache/kagglehub/datasets/kmader/rsna-bone-age/versions/2',
#                  resize_to=(256, 256), augment=False, preload=False, preloaded_data=None):
#         """
#         Given the root directory of the dataset, this function initializes the
#         data set

#         :param data_dir: List with paths of raw images
#         """

#         self._resize_to = resize_to
#         self._data_dir = data_dir
#         self._augment = augment
#         self._preload = preload

#         self._df = pd.read_csv(self._data_dir+f'/boneage-training-dataset.csv')

#         self._img_file_names = []
#         self._labels = []

#         for i in range(self._df.shape[0]):
#             # self._img_file_names.append(self._data_dir+f"/boneage-training-dataset/boneage-training-dataset/"
#             #                                            f"{self._df['id'][i]}.png")
#             self._img_file_names.append(self._data_dir+f"/boneage-training-dataset/"
#                                                        f"{self._df['id'][i]}.png")
#             self._labels.append(self._df['boneage'][i])

#         # normalize labels
#         self._labels = np.array(self._labels, dtype=np.float64)
#         self._labels = self._labels - self._labels.min()
#         self._labels = self._labels / self._labels.max()
#         self._labels = torch.tensor(self._labels).float().unsqueeze(-1)

#         self._imgs = []

#         if self._preload:
#             for fname in tqdm(self._img_file_names):
#                 x = io.imread(fname, as_gray=True)
#                 x = np.atleast_3d(x)
#                 max_size = np.max(x.shape)

#                 trans_always1 = [
#                     transforms.ToPILImage(),
#                     transforms.CenterCrop(max_size),
#                     transforms.Resize(self._resize_to),
#                 ]

#                 trans = transforms.Compose(trans_always1)
#                 x = trans(x)
#                 self._imgs.append(x)
#         else:
#             if preloaded_data:
#                 self._labels = preloaded_data[0]
#                 self._imgs = preloaded_data[1]
#                 self._preload = True

#     def __len__(self):
#         return len(self._img_file_names)

#     def __getitem__(self, idx):
#         if self._preload:
#             x = self._imgs[idx]
#             size = x.size
#         else:
#             x = io.imread(self._img_file_names[idx], as_gray=True)
#             x = np.atleast_3d(x)
#             max_size = np.max(x.shape)

#             trans_always1 = [
#                 transforms.ToPILImage(),
#                 transforms.CenterCrop(max_size),
#                 transforms.Resize(self._resize_to),
#             ]

#             trans = transforms.Compose(trans_always1)
#             x = trans(x)
#             w, h = x.size
#             size = (h, w)

#         y = self._labels[idx]

#         trans_augment = []
#         if self._augment:
#             trans_augment.append(transforms.RandomHorizontalFlip())
#             # trans_augment.append(transforms.RandomRotation(10, resample=PIL.Image.BILINEAR))
#             trans_augment.append(transforms.CenterCrop(size))
#             trans_augment.append(transforms.RandomCrop(size, padding=8))

#         mean = [0.14344494]
#         std = [0.18635063]

#         trans_always2 = [
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ]
#         trans = transforms.Compose(trans_augment+trans_always2)

#         x = trans(x)

#         return x, y


class ThreadSafeDict:
    def __init__(self):
        self._dict = {}
        self._lock = threading.Lock()

    def set(self, key, value):
        with self._lock:
            self._dict[key] = value

    def get(self, key):
        with self._lock:
            return self._dict.get(key)

    def remove(self, key):
        with self._lock:
            if key in self._dict:
                del self._dict[key]

    def __repr__(self):
        with self._lock:
            return repr(self._dict)
    
    def contains(self, key):
        """Check if a key is in the dictionary in a thread-safe manner."""
        with self._lock:
            return key in self._dict


IMGS = ThreadSafeDict()

def load_grayscale_image_from_zip(png_file_name):
    """
    Load a grayscale PNG image directly from a zip file.

    Parameters:
    zip_file_path (str): Path to the zip file.
    png_file_name (str): Name of the PNG file inside the zip.

    Returns:
    Image: Grayscale PIL Image.
    """
    zip_file_path = "/home/dsi/rotemnizhar/Downloads/rsna-bone-age.zip" 
    file_dir = "boneage-training-dataset/boneage-training-dataset"
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        with zip_ref.open(f"{file_dir}/{png_file_name}") as file:
            # Read the PNG file as a byte stream and convert to grayscale
            img = Image.open(file).convert('L')
            return img
        
        
class BoneAgeDataset(Dataset):
    """
    Loads the rsna bone age data set
    """

    def __init__(self, group, data_dir='/home/dsi/rotemnizhar/Downloads/rsna-bone-age/',
                 resize_to=(256, 256), augment=False):
        """
        Given the root directory of the dataset, this function initializes the
        data set

        :param data_dir: List with paths of raw images
        """

        self._resize_to = resize_to
        self._data_dir = data_dir
        self._augment = augment

        self._df = pd.read_csv(self._data_dir+f'/boneage-training-dataset.csv')

        self._img_file_names = []
        labels = []

        for i in range(self._df.shape[0]):
            # self._img_file_names.append(self._data_dir+f"/boneage-training-dataset/boneage-training-dataset/"
            #                                            f"{self._df['id'][i]}.png")
            self._img_file_names.append(self._data_dir+f"/boneage-training-dataset/"
                                                       f"{self._df['id'][i]}.png")
            labels.append(self._df['boneage'][i])

        # global IMGS
        # ignored_list = [2194]
        # if len(IMGS)==0:
        #     for fname in tqdm(self._img_file_names):
        #         idx = int(fname.split("/")[-1][:-4])
        #         if idx not in ignored_list:
        #             try:
        #                 x = io.imread(fname, as_gray=True)
        #                 IMGS[fname] = x
        #             except Exception as e:
        #                 print(f"file: {fname} will not be used, got an error: {e}")

        # normalize labels
        labels = np.array(labels, dtype=np.float64)
        labels = labels - labels.min()
        labels = labels / labels.max()
        self._labels = torch.tensor(labels).float().unsqueeze(-1)

        self._data__indices_dir = "/home/dsi/rotemnizhar/dev/regression_calibration/data_indices"
        indices = torch.load(f'{self._data__indices_dir}/boneage_{group}_indices.pth')
        # self._indices = []
        # for i in indices:
        #     if i in IMGS:
        #         self._indices.append(i)
        self._indices = indices
            
            


    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        global IMGS
        # x = IMGS[self._img_file_names[idx]]
        file_name = self._img_file_names[idx].split("/")[-1]
        file_dir = '/home/dsi/rotemnizhar/Downloads/rsna-bone-age/boneage-training-dataset/boneage-training-dataset'
        if IMGS.contains(file_name):
            x = IMGS.get(file_name)
        else:
            x = load_grayscale_image_from_zip(file_name)
            IMGS.set(file_name, x)
        # x = Image.open(f"{file_dir}/{file_name}").convert('L')
        x = np.atleast_3d(x)
        max_size = np.max(x.shape)

        trans_always1 = [
                transforms.ToPILImage(),
                transforms.CenterCrop(max_size),
                transforms.Resize(self._resize_to),
            ]

        trans = transforms.Compose(trans_always1)
        x = trans(x)
        w, h = x.size
        size = (h, w)

        y = self._labels[idx]

        trans_augment = []
        if self._augment:
            trans_augment.append(transforms.RandomHorizontalFlip())
            # trans_augment.append(transforms.RandomRotation(10, resample=PIL.Image.BILINEAR))
            trans_augment.append(transforms.CenterCrop(size))
            trans_augment.append(transforms.RandomCrop(size, padding=8))

        mean = [0.14344494]
        std = [0.18635063]

        trans_always2 = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        trans = transforms.Compose(trans_augment+trans_always2)

        x = trans(x)

        return x, y
