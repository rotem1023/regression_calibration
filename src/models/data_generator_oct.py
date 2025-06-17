import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
import PIL
from torchvision import transforms
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from glob import glob


class OCTDataset(Dataset):
    """
    Loads the OCT gastrointestinal dataset.
    """

    def __init__(self, group, resize_to=(256, 256), augment=False):
        """
        Given the root directory of the dataset, this function initializes the data set

        :param data_dir: List with paths of raw images
        """
        data_dir = "/home/dsi/rotemnizhar/dev/regression_calibration/src/models/data/oct/data/data"
        indexes_dir = "/home/dsi/rotemnizhar/dev/regression_calibration/data_indices"
        indexes_file = f"{indexes_dir}/oct_{group}_indices.pth"
        indices = torch.load(indexes_file)
        
        self._indices = indices
        self._resize_to = resize_to
        self._data_dir = data_dir
        self._augment = augment
        self._img_file_names = sorted(glob(data_dir + "/*.npz"))



        # max values of output for normalization
        self._max_vals = np.array([0.999612, 0.999535, 0.599804, 5.99884, 5.998696, 7.998165])



    def __len__(self):
        return len(self._indices)
    
    
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


    def __getitem__(self, idx):
        try:
            x, label = self._load_npz(self._img_file_names[self._indices[idx]])
        except Exception as e:
            print(f"Error loading file {self._img_file_names[self._indices[idx]]}: {e}")
            raise e
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