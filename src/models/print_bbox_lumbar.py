import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import io
import cv2
from PIL import Image, ImageDraw
import load_trained_models
import math
import torch
import numpy as np
import random
from cqr_model import BreastPathQModel


# Set fixed seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


Q_CP_X = {1: {'efficientnetb4': 1.7737019300460815}}
Q_CP_Y = {1: {'efficientnetb4': 1.7279503345489502}}
Q_NEW_X = {1: {'efficientnetb4': 0.07284029275178909}}
Q_NEW_Y = {1: {'efficientnetb4': 0.07355879656970502}}
Q_CQR_X = {1: {'efficientnetb4': 0.01565587818622589}}
Q_CQR_Y = {1: {'efficientnetb4': 0.09326351061463356}}


class Bbox():
    def __init__(self, x_minus, x_plus, y_minus, y_plus):
        self.x_minus = x_minus
        self.x_plus = x_plus
        self.y_minus = y_minus
        self.y_plus = y_plus
        
    def toList(self):
        return [self.x_minus, self.y_minus, self.x_plus, self.y_plus]


def get_cqr_model(base_model, level, alpha, dim, device):
    models_dir = f'/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots/cqr/one_dim/{dim}'
    checkpoint = torch.load(f'{models_dir}/{base_model}_lumbar_L{level}_alpha_{alpha}_cqr_best.pth.tar', map_location=device)
    model = BreastPathQModel(base_model, out_channels=2).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model


class LumbarDataset(Dataset):
    """
    Loads the EndoVis instrument tracking dataset with bounding boxes.
    """

    def __init__(self, level, mode='train', scale=1.0, augment=False):
        self._scale = scale
        self._level = level

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
                self._img_file_names = self._img_file_names[:int(n * 0.6)]
            elif mode == 'val':
                self._img_file_names = self._img_file_names[int(n * 0.6):int(n * 0.8)]
            else:
                self._img_file_names = self._img_file_names[int(n * 0.8):]
        
        
        base_model = 'efficientnetb4'
        base_model_dist = 'resnet50'
        assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
        self.device = torch.device("cuda:1")
        loss = 'gaussian'
        one_output = False
        lambda_param = 1
        level = 1
        alpha = 0.05
        self.model_x = load_trained_models.get_model_lumbar(base_model, level, None, self.device, loss=loss, pred_x=True, pred_y=False)
        self.dist_model_x = load_trained_models.get_model_lumbar(base_model_dist, level, base_model, self.device, lambda_param=lambda_param, one_out=one_output, loss=loss, pred_x=True, pred_y=False)
        self.model_y = load_trained_models.get_model_lumbar(base_model, level, None, self.device, loss=loss, pred_x=False, pred_y=True)
        self.dist_model_y = load_trained_models.get_model_lumbar(base_model_dist, level, base_model, self.device, lambda_param=lambda_param, one_out=one_output, loss=loss, pred_x=False, pred_y=True)
        self.cqr_model_x = get_cqr_model(base_model=base_model, level=level, alpha=0.05, dim = 'x', device=self.device)
        self.cqr_model_y = get_cqr_model(base_model=base_model, level=level, alpha=0.05, dim = 'y', device=self.device)
        self.q_x_my_model = Q_NEW_X[level][base_model]
        self.q_y_my_model = Q_NEW_Y[level][base_model]
        self.q_x_cp =  Q_CP_X[level][base_model]
        self.q_y_cp = Q_CP_Y[level][base_model]  
        self.q_x_cqr = Q_CQR_X[level][base_model]
        self.q_y_cqr = Q_CQR_Y[level][base_model]     
    
    def _calc_b_box(self, x):
        x = x.to(self.device).unsqueeze(0)
        mu_x, logvar_x, _ = self.model_x(x, dropout=False, mc_dropout=False, test=False)
        mu_y, logvar_y, _ = self.model_y(x, dropout=False, mc_dropout=False, test=False)
        
        mu_x, mu_y, logvar_x, logvar_y = mu_x.detach(), mu_y.detach(), logvar_x.detach(), logvar_y.detach()
        self.dist_model_x.eval()
        print(f"x: {x[0][0,2][0:5]}")
        distance_x =self.dist_model_x(x).detach()
        pos_dist_x, neg_dist_x= distance_x[:,0], distance_x[:,1]
        x_minus = mu_x - neg_dist_x - self.q_x_my_model
        x_plus = mu_x + pos_dist_x + self.q_x_my_model
        
        
        distance_y =self.dist_model_y(x).detach()
        pos_dist_y, neg_dist_y =  distance_y[:,0] , distance_y[:,1]
        y_minus = mu_y - neg_dist_y -self.q_y_my_model
        y_plus  = mu_y + pos_dist_y + self.q_y_my_model
        
        print(f"mu x: {mu_x}, mu y= {mu_y}, neg dist_x = {neg_dist_x}, pos dist x: {pos_dist_x}, neg dist y: {neg_dist_y}, pos dist y: {pos_dist_y}")
        
        bbox_my = Bbox(x_minus=x_minus.item(), x_plus=x_plus.item(), y_minus=y_minus.item(), y_plus=y_plus.item())
        
        sd_x, sd_y = math.sqrt(math.exp(logvar_x)), math.sqrt(math.exp(logvar_y))
        
        x_minus = mu_x - sd_x * self.q_x_cp
        x_plus = mu_x + sd_x * self.q_x_cp
        
        y_minus = mu_y - sd_y * self.q_y_cp
        y_plus = mu_y + sd_y * self.q_y_cp
        
        bbox_cp = Bbox(x_minus=x_minus.item(), x_plus=x_plus.item(), y_minus=y_minus.item(), y_plus=y_plus.item())
        
        
        # cqr 
        cqr_x = self.cqr_model_x(x).detach()
        cqr_y = self.cqr_model_y(x).detach()
        
        x_minus_cqr = cqr_x[0][0] - self.q_x_cqr
        x_plus_cqr = cqr_x[0][1] + self.q_x_cqr
        y_minus_cqr = cqr_y[0][0] - self.q_y_cqr
        y_plus_cqr = cqr_y[0][1] + self.q_y_cqr
        bbox_cqr = Bbox(x_minus=x_minus_cqr, x_plus=x_plus_cqr, y_minus=y_minus_cqr, y_plus=y_plus_cqr)

        
        return bbox_my, bbox_cp, bbox_cqr, mu_x, mu_y
        
            
            
            
            
    
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
        trans_always2 = [transforms.ToTensor()]
        trans = transforms.Compose(trans_always2)
        print(f"file: {self._img_file_names[idx]}")
        filename_values = self._img_file_names[idx]
        split_val = filename_values.split(',')
        y = np.array(split_val[1:], dtype=np.float32)

        img_path = os.path.join(self.data_dir, split_val[0])
        x = io.imread(img_path)
        x = np.atleast_3d(x)

        # Convert grayscale to RGB
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)

        # Convert to PIL and resize
        x_calc = self.to_pil_and_resize(x, self._scale)

        # Extract bounding box coordinates from file (assumed format: x_min, y_min, x_max, y_max)
        my_bbox, cp_bbbox, cqr_bbox, mu_x, mu_y = self._calc_b_box(trans(x_calc))

        # Draw bounding box
        mu = [mu_x, mu_y]
        
        my_image = self.draw_bounding_box(x_calc, my_bbox, y, mu)
        # my_image.save(f"{img_path.split('/')[-1]}_my_image.jpg")
        
        cqr_image = self.draw_bounding_box(x_calc, cqr_bbox, y, mu, outline_color='pink')
       
        cp_image = self.draw_bounding_box(x_calc, cp_bbbox, y, mu, outline_color='orange')
        cp_image.save(f"{img_path.split('/')[-1]}_cp_image.jpg")
        
        
        
        x = trans(x)

        return x

    
    def draw_bounding_box(self, image, bbox, true_value, predicted_value, outline_color = 'red'):
        """
        Draws a bounding box and two points (blue & green) on the given PIL image.

        :param image: PIL Image
        :param bbox: Bounding box with relative values [x_min, y_min, x_max, y_max] (0-1 range)
        :param point1: First point (x, y) in relative coordinates (0-1), will be blue.
        :param point2: Second point (x, y) in relative coordinates (0-1), will be green.
        :return: PIL Image with bounding box and points drawn.
        """
        draw = ImageDraw.Draw(image)
        
        # Get image width and height
        img_width, img_height = image.size

        # Convert relative bbox coordinates to absolute pixels
        x_min = int(bbox.x_minus * img_width)
        y_min = int(bbox.y_minus * img_height)
        x_max = int(bbox.x_plus * img_width)
        y_max = int(bbox.y_plus * img_height)

        # Draw the bounding box (red)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=outline_color, width=3)

        # Convert relative points to absolute pixels
        true_x = int(true_value[0] * img_width)
        true_y = int(true_value[1] * img_height)
        predict_x = int(predicted_value[0] * img_width)
        predict_y = int(predicted_value[1] * img_height)

        # Draw points (small circles)
        point_radius = 5  # Size of the points
        draw.ellipse([true_x - point_radius, true_y - point_radius, true_x + point_radius, true_y + point_radius], fill="blue")
        draw.ellipse([predict_x - point_radius, predict_y - point_radius, predict_x + point_radius, predict_y + point_radius], fill="green")

        return image


dataset = LumbarDataset(level=1, mode='test', scale=1.0, augment=False)

for i in range(20):
    sample = dataset[i]