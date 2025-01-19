import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import os


class LumbarDataset(Dataset):
    """
    Loads the EndoVis instrument tracking data set
    """

    def __init__(self, level, mode='train', scale=1.0, augment=False):
        """
        Given the root directory of the dataset, this function initializes the
        data set

        :param data_dir: List with paths of raw images
        """

        self._scale = scale
        self._level = level
        self._augment = augment
        
        # get current file dir
        current_dir = os.path.dirname(os.path.abspath(__file__))
        idx_dir = os.path.join(current_dir, 'data')
        self.data_dir = idx_dir
        idx_file = os.path.join(idx_dir, f'L{level}.txt')
        # read the index file
        with open(idx_file, 'r') as f:
            self._img_file_names = f.readlines()
            n = len(self._img_file_names)
            self._img_file_names = [x.strip() for x in self._img_file_names]
            # split the data into train, val, test
            if mode == 'train':
                self._img_file_names = self._img_file_names[:int(n*0.6)]
            elif mode == 'val':
                self._img_file_names = self._img_file_names[int(n*0.6):int(n*0.8)]
            else:
                self._img_file_names = self._img_file_names[int(n*0.8):]


    @staticmethod
    def to_pil_and_resize(x, scale):
        w, h, _ = x.shape
        new_size = (int(w * scale), int(h * scale))
        target_size = (224, 224)

        trans_always1 = [
            transforms.ToPILImage(),
            transforms.Resize(target_size),
        ]

        trans = transforms.Compose(trans_always1)
        x = trans(x)
        return x

    def __len__(self):
        return len(self._img_file_names)

    def __getitem__(self, idx):
        filename_values = self._img_file_names[idx]
        split_val = filename_values.split(',')
        img_path = os.path.join(self.data_dir,split_val[0])
        x = io.imread(img_path)
        x = np.atleast_3d(x)
        if x.shape[2] == 1:  # Check if the image is grayscale
            x = np.repeat(x, 3, axis=2)
        x = self.to_pil_and_resize(x, self._scale)

        y = np.array(split_val[1:], dtype=np.float32)
        y = (y[0]+ y[1]/2)

        # horizontal flipping
        # if self._augment and np.random.rand() > 0.5:
        #     x = transforms.functional.hflip(x)
        #     y[1] = 1 - y[1]

        trans_augment = []
        # if self._augment:
        #     trans_augment.append(transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2,
        #                                                                         saturation=0.2, hue=0.1)], p=0.5))

        # trans_always2 = [
        #     transforms.ToTensor(),
        # ]
        # trans = transforms.Compose(trans_augment+trans_always2)
        if self._augment:
            trans_augment.append(transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                                saturation=0.2, hue=0.1)], p=0.5))

        trans_always2 = [
            transforms.ToTensor(),
        ]
        trans = transforms.Compose(trans_augment + trans_always2)

        x = trans(x)

        return x, y


def demo():
    from matplotlib import pyplot as plt

    dataset_train = EndoVisDataset(data_dir='/media/data/EndoVis15_instrument_tracking/test',
                                   augment=False, scale=0.5, preload=True)
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)

    print("Train dataset length:", len(data_loader_train))

    for i_batch, b in enumerate(data_loader_train):
        x, y = b
        h, w = (x.size(2), x.size(3))
        print(i_batch, y, x.size(), y.size(), x.type(), y.type())
        print(y[0, 0]*w, y[0, 1]*h)
        plt.subplot(1, 1, 1)
        plt.imshow(x.data.cpu().numpy()[0, 0])
        plt.plot(y[0, 0]*w, y[0, 1]*h, 'rx')

        # plt.show()

        plt.pause(0.1)
        ret = plt.waitforbuttonpress(0.1)
        if ret:
            break

        plt.clf()


def perf_test():
    dataset_train = EndoVisDataset(data_dir='/media/data/EndoVis15_instrument_tracking/train',
                                   augment=False, scale=0.5, preload=True)
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)

    print("Train dataset length:", len(data_loader_train))

    for b in tqdm(data_loader_train):
        x, y = b


def calc_mean_std():
    dataset = EndoVisDataset(data_dir='/media/fastdata/laves/rsna-bone-age/', augment=False, preload=False)
    data_loader = DataLoader(dataset, batch_size=1)

    accu = []

    for data, _ in tqdm(data_loader):
        accu.append(data.data.cpu().numpy().flatten())

    accu = np.concatenate(accu)

    return accu.mean(), accu.std()


if __name__ == "__main__":
    # mean, std = calc_mean_std()
    # print("mean =", mean)
    # print("std =", std)
    demo()
    # perf_test()
