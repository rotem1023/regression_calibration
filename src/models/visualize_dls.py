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
from models import BreastPathQModel
from cqr_model import BreastPathQModel as BreastPathQModelCqr
from data_generator_lumbar import LumbarDataset as LumbarDatasetOriginal


# Set fixed seed for reproducibility
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

Q_CP_DIM = {1 : {'efficientnetb4':2.340525758266449}}
Q_CP_X = {1: {'efficientnetb4': 1.4532232463359833}}
Q_CP_Y = {1: {'efficientnetb4': 1.339308488368988}}
Q_CQR_X = {1: {'efficientnetb4': 0.03720600605010986}}
Q_CQR_Y = {1: {'efficientnetb4': 0.024203389883041382}}
Q_CQR_MAX_RANK_X = {1: {'efficientnetb4': 0.03660971522331238}}
Q_CQR_MAX_RANK_Y = {1: {'efficientnetb4': 0.023421835899353028}}


class Bbox():
    def __init__(self, x_minus, x_plus, y_minus, y_plus):
        self.x_minus = x_minus
        self.x_plus = x_plus
        self.y_minus = y_minus
        self.y_plus = y_plus
        
    def toList(self):
        return [self.x_minus, self.y_minus, self.x_plus, self.y_plus]
    
class Elipsoid():
    def __init__(self, x, y, a, b):
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        
    def toList(self):
        return [self.x, self.y, self.a, self.b]


def get_cqr_model(base_model, level, alpha, device):
    models_dir = f'/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots/cqr'
    checkpoint = torch.load(f'{models_dir}/{base_model}_lumbar_L{level}_alpha_{alpha}_cqr_dims_best.pth.tar', map_location=device) 
    model = BreastPathQModelCqr(base_model, out_channels=2).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model


class LumbarDataset(Dataset):
    """
    Loads the EndoVis instrument tracking dataset with bounding boxes.
    """

    def __init__(self, level, mode='train', scale=0.5, augment=False):
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
        assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
        self.device = torch.device("cuda:1")
        level = 1
        alpha = 0.05
        checkpoint = torch.load(f'/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots/{base_model}_gaussian_lumbar_L{level}_snapshot_dims.pth.tar', map_location=self.device)
        self.model_cp = BreastPathQModel(base_model, out_channels=2).to(self.device)
        self.model_cp.load_state_dict(checkpoint['state_dict'])
        print(f"epoch: {checkpoint['epoch']}")
        # self.model_cp.eval()
        self.model_cqr = get_cqr_model(base_model, level, alpha, device=self.device)
        # self.model_cqr.eval()
        self.q_x_cp =  Q_CP_X[level][base_model]
        self.q_y_cp = Q_CP_Y[level][base_model]  
        self.q_x_cqr = Q_CQR_X[level][base_model]
        self.q_y_cqr = Q_CQR_Y[level][base_model] 
        self.q_cp = Q_CP_DIM[level][base_model]    
        self.cp_bonf_areas = []
        self.cqr_areas = []
        self.original_dataset = LumbarDatasetOriginal(level=level, mode=mode, scale=scale, augment=augment)
    
    def _get_mu_sd(self, mu, logvar):
        mu = mu.clamp(0, 1).permute(1, 0, 2)
        mu = mu.mean(dim=1).cpu()
        logvar = logvar.permute(1, 0, 2)
        logvar = logvar.mean(dim=1).cpu()
        var = logvar.exp()
        sd = var.sqrt()
        return mu, sd
    
    def create_final_preds(self, t_p_s):
        preds = []
        for i in range(len(t_p_s[0])): # dims
            dim_preds = []
            for j in range(len(t_p_s)): # number o epochs
                dim_preds.append(t_p_s[j][i].detach().cpu())
            dim_preds = torch.cat(dim_preds, dim=1).clamp(0, 1).permute(1,0,2).mean(dim=1)
            preds.append(dim_preds)
        x_preds = preds[0]
        y_preds = preds[1]
        return x_preds, y_preds


    def create_final_preds(self, t_p_s):
        preds = []
        for i in range(len(t_p_s[0])):
            dim_preds = []
            for j in range(len(t_p_s)):
                dim_preds.append(t_p_s[j][i].detach().cpu())
            dim_preds = torch.cat(dim_preds, dim=1).clamp(0, 1).permute(1,0,2).mean(dim=1)
            preds.append(dim_preds)
        return torch.stack(preds)

    def create_final_preds_oct(self, t_p_s):
        preds = self.create_final_preds(t_p_s)
        preds = preds.permute(2, 1, 0)
        x_preds = preds[0]
        y_preds = preds[1]
        return x_preds, y_preds
    
    def clac_cqr_limits(self, x):
        x = x.to(self.device)
        preds = self.model_cqr(x, dropout=True, mc_dropout=True, test=True)
        final_preds = self.create_final_preds_oct([preds])
        return final_preds
    
    def clac_limits(self, x):
        x = x.to(self.device).unsqueeze(0)
        with torch.no_grad():
            mu, logvar, _ = self.model_cp(x, dropout=True, mc_dropout=True, test=True)
        mu, sd = self._get_mu_sd(mu, logvar)
        mu_x = mu[0][0]
        mu_y = mu[0][1]
        sd_x = sd[0][0]
        sd_y = sd[0][1]
        bonf_x_minus = mu_x - self.q_x_cp * sd_x
        bonf_x_plus = mu_x + self.q_x_cp * sd_x
        bonf_y_minus = mu_y - self.q_y_cp * sd_y
        bonf_y_plus = mu_y + self.q_y_cp * sd_y
        cp_bonf_area = (bonf_x_plus - bonf_x_minus) * (bonf_y_plus - bonf_y_minus)
        print("area bonf:", cp_bonf_area.item())
        bbox_bonf_cp = Bbox(x_minus=bonf_x_minus.item(), x_plus=bonf_x_plus.item(), y_minus=bonf_y_minus.item(), y_plus=bonf_y_plus.item())
        with torch.no_grad():
            x_cqr_preds, y_cqr_preds = self.clac_cqr_limits(x)
        # x_cqr_preds = [[0][0]]
        # y_cqr_preds = [[0][0]]
        x_minus_cqr = x_cqr_preds[0][0]  - self.q_x_cqr
        x_plus_cqr = x_cqr_preds[0][1] + self.q_x_cqr
        y_minus_cqr = y_cqr_preds[0][0]  - self.q_y_cqr
        y_plus_cqr = y_cqr_preds[0][1] + self.q_y_cqr
        cqr_area = (x_plus_cqr - x_minus_cqr) * (y_plus_cqr - y_minus_cqr)
        print("area cqr:", cqr_area.item())
        bbox_cqr = Bbox(x_minus=x_minus_cqr, x_plus=x_plus_cqr, y_minus=y_minus_cqr, y_plus=y_plus_cqr)
        
        dist_x = sd_x * math.sqrt(self.q_cp)
        dist_y = sd_y * math.sqrt(self.q_cp)
        elipsoid_cp = Elipsoid(x=mu_x.item(), y=mu_y.item(), a=dist_x.item(), b=dist_y.item())

        self.cp_bonf_areas.append(cp_bonf_area.item())
        self.cqr_areas.append(cqr_area.item())
        
        return elipsoid_cp, bbox_bonf_cp, bbox_cqr, mu_x, mu_y
        
            
            
            
            
    
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
        # trans_always2 = [transforms.ToTensor()]
        # trans = transforms.Compose(trans_always2)
        print(f"file: {self._img_file_names[idx]}")
        filename_values = self._img_file_names[idx]
        split_val = filename_values.split(',')
        # y = np.array(split_val[1:], dtype=np.float32)

        img_path = os.path.join(self.data_dir, split_val[0])
        # x = io.imread(img_path)
        # x = np.atleast_3d(x)

        # # Convert grayscale to RGB
        # if x.shape[2] == 1:
        #     x = np.repeat(x, 3, axis=2)

        # # Convert to PIL and resize
        # x_calc = self.to_pil_and_resize(x, self._scale)
        
        x_calc, y = self.original_dataset.__getitem__(idx)

        # Extract bounding box coordinates from file (assumed format: x_min, y_min, x_max, y_max)
        cp_dim_elip, cp_bonf_bbox, cqr_bonf_bbox, mu_x, mu_y = self.clac_limits(x_calc)

        # Draw bounding box
        mu = [mu_x, mu_y]
        
        # my_image = self.draw_bounding_box(x_calc, cp_dim_elip, y, mu)
        # # my_image.save(f"{img_path.split('/')[-1]}_my_image.jpg")
        
        # cqr_image = self.draw_bounding_box(x_calc, cqr_bbox, y, mu, outline_color='pink')
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(x_calc)
        cp_image = self.draw_elipsoid(pil_image, cp_dim_elip, outline_color='red')
        # cp_image = self.draw_bounding_box(pil_image, cp_bonf_bbox, y, mu, outline_color='orange')
        cp_image = self.draw_bounding_box(pil_image, cqr_bonf_bbox, y, mu, outline_color='pink')
        cp_image.save(f"{img_path.split('/')[-1]}_cp_image.jpg")
        
        
        

        return x_calc

    
    def draw_elipsoid(self, image, elipsoid, outline_color='red'):
        draw = ImageDraw.Draw(image)
        img_width, img_height = image.size
        x_scaled = int(elipsoid.x * img_width)
        y_scaled = int(elipsoid.y * img_height)
        a_scaled = int(elipsoid.a * img_width)
        b_scaled = int(elipsoid.b * img_height)
        
        draw.ellipse((x_scaled - a_scaled, y_scaled - b_scaled,
                      x_scaled + a_scaled, y_scaled + b_scaled),
                     outline=outline_color, width=3)
        return image
 
 
 
def draw_elipsoid( image, elipsoid, outline_color='red'):
        draw = ImageDraw.Draw(image)
        img_width, img_height = image.size
        x_scaled = int(elipsoid.x * img_width)
        y_scaled = int(elipsoid.y * img_height)
        a_scaled = int(elipsoid.a * img_width)
        b_scaled = int(elipsoid.b * img_height)
        
        draw.ellipse((x_scaled - a_scaled, y_scaled - b_scaled,
                      x_scaled + a_scaled, y_scaled + b_scaled),
                     outline=outline_color, width=3)
        return image
           
def draw_bounding_box(image, bbox, outline_color = 'red'):
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

        return image

def draw_center_point(image, point, color='blue'):
    """
    Draws a center point on the given PIL image.

    :param image: PIL Image
    :param point: Center point (x, y) in relative coordinates (0-1)
    :param color: Color of the point
    :return: PIL Image with center point drawn.
    """
    draw = ImageDraw.Draw(image)
    
    # Get image width and height
    img_width, img_height = image.size

    # Convert relative point coordinates to absolute pixels
    x = int(point[0] * img_width)
    y = int(point[1] * img_height)

    # Draw the center point (blue)
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

    return image

def darw_and_save_image(image, cp_elipsoid, bbox_bonf_cp, bbox_cqr, bb_cqr_max_rank, true_value, predicted_value, save_path):
    cp_image = draw_elipsoid(image, cp_elipsoid, outline_color='red')
    # cp_image = draw_bounding_box(image, bbox_bonf_cp, outline_color='orange')
    cp_image = draw_bounding_box(image, bbox_cqr, outline_color='pink')
    # cp_image = draw_bounding_box(image, bb_cqr_max_rank, outline_color='purple')
    cp_image = draw_center_point(cp_image, true_value, color='blue')
    cp_image = draw_center_point(cp_image, predicted_value, color='green')
    cp_image.save(save_path)
    return is_point_in_ellipse(true_value[0].item(), true_value[1].item(), cp_elipsoid.x, cp_elipsoid.y, cp_elipsoid.a, cp_elipsoid.b)

def draw_image(index, x, mu, sd, target, lower_pred_x, lower_pred_y, higher_pred_x, higher_pred_y, base_model='efficientnetb4', dataset='lumbar', level=1, mode='test', alpha=0.05):
    bonf_x_minus = mu[0] - Q_CP_X[level][base_model] * sd[0]
    bonf_x_plus = mu[0] + Q_CP_X[level][base_model] * sd[0]
    bonf_y_minus = mu[1] - Q_CP_Y[level][base_model] * sd[1]
    bonf_y_plus = mu[1] + Q_CP_Y[level][base_model] * sd[1]
    cp_bonf_area = (bonf_x_plus - bonf_x_minus) * (bonf_y_plus - bonf_y_minus)
    print("area bonf:", cp_bonf_area.item())
    bbox_bonf_cp = Bbox(x_minus=bonf_x_minus.item(), x_plus=bonf_x_plus.item(), y_minus=bonf_y_minus.item(), y_plus=bonf_y_plus.item())

    dist_x = sd[0] * math.sqrt(Q_CP_DIM[level][base_model])
    dist_y = sd[1] * math.sqrt(Q_CP_DIM[level][base_model])
    elipsoid_cp = Elipsoid(x=mu[0].item(), y=mu[1].item(), a=dist_x.item(), b=dist_y.item())
    
    x_minus_cqr = lower_pred_x - Q_CQR_X[level][base_model]
    x_plus_cqr = higher_pred_x + Q_CQR_X[level][base_model]
    y_minus_cqr = lower_pred_y  - Q_CQR_Y[level][base_model]
    y_plus_cqr = higher_pred_y + Q_CQR_Y[level][base_model]
    cqr_area = (x_plus_cqr - x_minus_cqr) * (y_plus_cqr - y_minus_cqr)
    print("area cqr:", cqr_area.item())
    bbox_cqr = Bbox(x_minus=x_minus_cqr, x_plus=x_plus_cqr, y_minus=y_minus_cqr, y_plus=y_plus_cqr)    
    
    
    x_minus_cqr_max_rank = lower_pred_x - Q_CQR_MAX_RANK_X[level][base_model]
    x_plus_cqr_max_rank = higher_pred_x + Q_CQR_MAX_RANK_X[level][base_model]
    y_minus_cqr_max_rank = lower_pred_y  - Q_CQR_MAX_RANK_Y[level][base_model]
    y_plus_cqr_max_rank = higher_pred_y + Q_CQR_MAX_RANK_Y[level][base_model]
    cqr_area_max_rank = (x_plus_cqr_max_rank - x_minus_cqr_max_rank) * (y_plus_cqr_max_rank - y_minus_cqr_max_rank)
    print("area cqr max rank:", cqr_area_max_rank.item())
    bbox_cqr_max_rank = Bbox(x_minus=x_minus_cqr_max_rank, x_plus=x_plus_cqr_max_rank, y_minus=y_minus_cqr_max_rank, y_plus=y_plus_cqr_max_rank)
    
    
    x = (np.transpose(x, (1, 2, 0)) * 255).astype(np.uint8)  # [3, 224, 224] -> [224, 224, 3]

    # Create PIL image
    image = Image.fromarray(x)
    return darw_and_save_image(image, elipsoid_cp, bbox_bonf_cp, bbox_cqr, bbox_cqr_max_rank,
                        true_value=target, predicted_value=mu,
                        save_path=f'/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results/visuals/{dataset}_dataset_model_{base_model}_level{level}_{mode}_alpha_{alpha}_index{index}.jpg'
                        )
    
    
def is_point_in_ellipse(x, y, h, k, a, b, theta=0):
    """
    Check if a 2D point (x, y) is inside or on an ellipse.
    
    Parameters:
        x, y   : point coordinates
        h, k   : ellipse center
        a      : semi-major axis
        b      : semi-minor axis
        theta  : rotation angle of the ellipse in radians (default = 0, aligned with axes)
    
    Returns:
        True if point is inside or on the ellipse, False otherwise.
    """
    # Translate point relative to ellipse center
    dx = x - h
    dy = y - k
    
    # Rotate point by -theta
    x_rot = dx * math.cos(theta) + dy * math.sin(theta)
    y_rot = -dx * math.sin(theta) + dy * math.cos(theta)
    
    # Check ellipse equation
    value = (x_rot**2) / (a**2) + (y_rot**2) / (b**2)
    return value <= 1


def draw_images(base_model='efficientnetb4', dataset = 'lumbar', level=1, mode='test', alpha = 0.05):
    # load data
    results_cp_dir = "/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results/predictions/dims"
    cp_file = f'{results_cp_dir}/{dataset}_dataset_model_{base_model}_level{level}_test_visual.pth.tar'
    results_cqr_dir = "/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results/predictions/cqr/dims"
    cqr_file = f'{results_cqr_dir}/{dataset}_dataset_cqr_model_{base_model}_alpha_{alpha}_level_{level}_test_visual.pth.tar'
    cp_data = torch.load(cp_file, map_location='cpu')
    cqr_data = torch.load(cqr_file, map_location='cpu')
    x_cp = cp_data['x']
    mu_cp = cp_data['mu']
    sd_cp = cp_data['sd']
    # target_cp = cp_data['target']
    x_cqr = cqr_data['x']
    lower_preds_x = cqr_data['lower_preds_x']
    lower_preds_y = cqr_data['lower_preds_y']
    higher_preds_x = cqr_data['higher_preds_x']
    higher_preds_y = cqr_data['higher_preds_y']
    target_cqr = cqr_data['target']
    target_cp = cp_data['target']
    in_elips = 0
    for i in range(len(mu_cp)):
        cur_x = x_cp[i]
        cur_mu = mu_cp[i]
        cur_sd = sd_cp[i]
        cqr_target = target_cp[i]
        cqr_lower_preds_x = lower_preds_x[i]
        cqr_lower_preds_y = lower_preds_y[i]
        cqr_higher_preds_x = higher_preds_x[i]
        cqr_higher_preds_y = higher_preds_y[i]
        in_elips += draw_image(i,cur_x, cur_mu, cur_sd, cqr_target, cqr_lower_preds_x, cqr_lower_preds_y, cqr_higher_preds_x, cqr_higher_preds_y, base_model=base_model, dataset=dataset, level=level, mode=mode, alpha=alpha)
    print(f"num in elips: {in_elips} out of {len(mu_cp)}")
    print(f"percent in elips: {in_elips/len(mu_cp)}")
    return        



draw_images()

dataset = LumbarDataset(level=1, mode='test', scale=1.0, augment=False)

for i in range(40):
    sample = dataset[i]

print(f"cp bonf avg size: {np.mean(dataset.cp_bonf_areas)}")
print(f"cqr bonf avg size {np.mean(dataset.cqr_areas)}")