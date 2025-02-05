import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
import PIL
from torchvision import transforms
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from glob import glob
from ThreadSafeDict import ThreadSafeDict

class OCTDataset(Dataset):
    """
    Loads the OCT gastrointestinal dataset.
    """

    def __init__(self, data_dir, resize_to=(256, 256), augment=False, preload=False, preloaded_data_from=None):
        """
        Given the root directory of the dataset, this function initializes the data set

        :param data_dir: List with paths of raw images
        """

        self._resize_to = resize_to
        self._data_dir = data_dir
        self._augment = augment
        self._preload = preload

        # max values of output for normalization
        self._max_vals = np.array([0.999612, 0.999535, 0.599804, 5.99884, 5.998696, 7.998165])

        if not preloaded_data_from:
            self._img_file_names = sorted(glob(data_dir + "/*.npz"))

            self._imgs = []  # list of PILs or empty
            self._labels = []  # list of normalized x,y pixel coordinates of tool base

            if self._preload:
                for fname in tqdm(self._img_file_names):
                    img, label = self._load_npz(fname)
                    img = self._argmax_project(img)
                    img = self._to_pil_and_resize(img, self._resize_to)
                    self._imgs.append(img)
                    self._labels.append(label/self._max_vals)
        else:
            if preloaded_data_from:
                self._labels = preloaded_data_from._labels
                self._imgs = preloaded_data_from._imgs
                self._img_file_names = preloaded_data_from._img_file_names
                self._resize_to = preloaded_data_from._resize_to
                self._preload = True

    @staticmethod
    def _to_pil_and_resize(x, new_size):
        trans_always1 = [
            transforms.ToPILImage(),
            transforms.Resize(new_size, interpolation=1),
        ]

        trans = transforms.Compose(trans_always1)
        x = trans(x)
        return x

    @staticmethod
    def _argmax_project(x):
        y = [np.argmax(x, axis=0), np.argmax(x, axis=1), np.argmax(x, axis=2)]
        return np.stack(y, axis=-1).astype(np.uint8)

    @staticmethod
    def _load_npz(file_name, rescale=True):
        f = np.load(file_name)
        img = f['data']
        pos = f['pos']

        img = img[8:]  # crop top 8 rows
        min_shape = np.min(img.shape)
        if rescale:
            img = zoom(img,
                       zoom=(min_shape / img.shape[0],
                             min_shape / img.shape[1],
                             min_shape / img.shape[2]),
                       order=0)

        img = img.transpose(2, 0, 1)  # permute data as it is in FORTRAN order

        return img, pos

    def __len__(self):
        return len(self._img_file_names)

    def __getitem__(self, idx):
        if self._preload:
            x = self._imgs[idx]
            y = np.array(self._labels[idx], dtype=np.float32)
        else:
            x, label = self._load_npz(self._img_file_names[idx])
            label = label/self._max_vals
            x = self._argmax_project(x)
            x = self._to_pil_and_resize(x, self._resize_to)
            y = np.array(label, dtype=np.float32)

        trans_augment = []
        if self._augment:
            trans_augment.append(transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                                saturation=0.2, hue=0.1)], p=0.5))

        trans_always2 = [
            transforms.ToTensor(),
        ]
        trans = transforms.Compose(trans_augment + trans_always2)

        x = trans(x)

        return x, y


def to_pil_and_resize(x, new_size):
    trans_always1 = [
            transforms.ToPILImage(),
            transforms.Resize(new_size, interpolation=1),
    ]

    trans = transforms.Compose(trans_always1)
    x = trans(x)
    return x


def argmax_project(x):
    y = [np.argmax(x, axis=0), np.argmax(x, axis=1), np.argmax(x, axis=2)]
    return np.stack(y, axis=-1).astype(np.uint8)

def load_npz(file_name, rescale=True):
    f = np.load(file_name)
    img = f['data']
    pos = f['pos']

    img = img[8:]  # crop top 8 rows
    min_shape = np.min(img.shape)
    if rescale:
        img = zoom(img,
                       zoom=(min_shape / img.shape[0],
                             min_shape / img.shape[1],
                             min_shape / img.shape[2]),
                       order=0)

    img = img.transpose(2, 0, 1)  # permute data as it is in FORTRAN order

    return img, pos

def calc_mean_std():
    dataset = OCTDataset(data_dir='/media/data/OBRDataset/OBRDataset', augment=False, preload=False)
    data_loader = DataLoader(dataset, batch_size=1)

    accu = []

    for data, _ in tqdm(data_loader):
        accu.append(data.data.cpu().numpy().flatten())

    accu = np.concatenate(accu)

    return accu.mean(), accu.std()



class OCTDataset(Dataset):
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
        
        # max values of output for normalization
        self._max_vals = np.array([0.999612, 0.999535, 0.599804, 5.99884, 5.998696, 7.998165])
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


        # normalize labels
        labels = np.array(labels, dtype=np.float64)
        labels = labels - labels.min()
        labels = labels / labels.max()
        self._labels = torch.tensor(labels).float().unsqueeze(-1)

        self._data__indices_dir = "/home/dsi/rotemnizhar/dev/regression_calibration/data_indices"
        indices = torch.load(f'{self._data__indices_dir}/boneage_{group}_indices.pth')

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
            x, label = load_npz(self._img_file_names[idx])
            label = label/self._max_vals
            x = argmax_project(x)
            x = to_pil_and_resize(x, self._resize_to)
            y = np.array(label, dtype=np.float32)
            IMGS.set(file_name, x)
        trans_augment = []
        if self._augment:
            trans_augment.append(transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                                saturation=0.2, hue=0.1)], p=0.5))

        trans_always2 = [
            transforms.ToTensor(),
        ]
        trans = transforms.Compose(trans_augment + trans_always2)

        x = trans(x)

        return x, y

