import numpy as np
import os
import sys
np.random.seed(1)
import torch
torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import SubsetRandomSampler, ConcatDataset, Subset
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from data_generator_endovis import EndoVisDataset
from data_generator_lumbar import LumbarDataset
from data_generator_oct import OCTDataset
from models import BreastPathQModel, DistancePredictor
from glob import glob
import statistics
import math
from data_generator_brain import BrainDatasetTest, BrainDatasetVal 
from functools import reduce
from operator import mul
from max_rank import adjusted_q_max_rank


    
def calc_bonferroni_q(target_calib, mu_calib, sd_calib, alpha): 
    qs = []
    ndims = target_calib.shape[1]
    updated_alpha = alpha / ndims
    for i in range(ndims):
        target_i = target_calib[:, i] 
        mu_i = mu_calib[:, i]
        sd_calib_i = sd_calib[:, i]
        q_dim = calc_optimal_q(target_i.unsqueeze(-1), mu_i.unsqueeze(-1), sd_calib_i.unsqueeze(-1), updated_alpha)
        qs.append(q_dim)
    return qs

def compute_in_range_len_one_dim(y_test, y_lower, y_upper):
    """ Compute average coverage and length of prediction intervals

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)

    Returns
    -------

    coverage : float, average coverage
    avg_length : float, average length

    """
    y_test = y_test.unsqueeze(1).mean(dim=1)
    in_the_range = torch.sum((y_test >= y_lower) & (y_test <= y_upper))
    length = abs(y_upper - y_lower)
    return ((y_test >= y_lower) & (y_test <= y_upper)), length

def compute_coverage_size_bonf(targets, preds, sds, q_s):
    coverages, lengths = [], []
    ndim = targets.shape[1]
    for i in range(ndim):
        cur_tgrets = targets[:, i]
        cur_preds = preds[:, i]
        cur_sd = sds[:, i]
        cur_q = q_s[i]
        y_lower = cur_preds - cur_sd * cur_q
        y_upper = cur_preds + cur_sd * cur_q
        cur_coverage, cur_length = compute_in_range_len_one_dim(cur_tgrets, y_lower, y_upper)
        coverages.append(cur_coverage)
        lengths.append(cur_length)

    overall_cvg = torch.stack(coverages).all(dim=0)

    # Convert boolean result to int (0/1)
    overall_cvg = overall_cvg.to(dtype=torch.uint8)   
    
    coverage = torch.sum(overall_cvg) / len(overall_cvg)
    length = reduce(mul, lengths)
    return  torch.mean(length).item(), coverage * 100
    

def calc_optimal_q_max_rank(target_calib, mu_calib, sd_calib, alpha):
    delta = torch.abs(target_calib - mu_calib)

    inv_cov_diag = 1.0 / (sd_calib ** 2)
    
    s_t = torch.sqrt((delta * inv_cov_diag * delta))
    
    s_t = s_t.cpu().numpy()
    
    return adjusted_q_max_rank(s_t, alpha)

# CP


def calc_optimal_q(target_calib, mu_calib, sd_calib, alpha, gc=False):
    delta = torch.abs(target_calib - mu_calib)

    # Compute Σ(x)^(-1) under diagonal assumption → just 1 / sd
    inv_cov_diag = 1.0 / (sd_calib ** 2)  # [N, D]
    ind_sd = (1.0 / sd_calib)

    # Compute Mahalanobis distance: (delta^T @ Σ^{-1} @ delta), simplified for diagonal
    if sd_calib.shape[-1] != 1:
        s_t = torch.sum(delta * inv_cov_diag * delta, dim=1)
    else: 
        s_t = torch.sum(delta * ind_sd, dim=1)  # [N, D] -> [N]
    # if sd_calib.shape[-1] == 1:
    #     s_t_tmp = torch.abs(target_calib-mu_calib) / sd_calib
    # s_t = torch.sum(torch.abs(target_calib-mu_calib), dim =1) / sd_calib
    if gc:
        S = (s_t).mean().sqrt()
        if alpha == 0.1:
            q = 1.64485 * S.item()
        elif alpha == 0.05:
            q = 1.95996 * S.item()
        else:
            print("Choose another value of alpha!! (0.1 / 0.05)")
    else:
        s_t_sorted, _ = torch.sort(s_t, dim=0)
        q_index = math.ceil((len(s_t_sorted)) * (1 - alpha))
        q = s_t_sorted[q_index].item()   
    return q

# CP/GC prediction

def calc_stats(q, target, mu, sd):
    dist_true = torch.sum(torch.abs(target - mu), dim =1) 
    dist_pred = math.sqrt(q) * sd
    area = avg_ellipsoid_volume(dist_pred)
    coverage = avg_cov_ellipsoid(q, target, mu, sd)
    in_elps = 0
    for i in range(len(dist_pred)):
        in_elps += is_point_in_ellipse(target[i,0].item(), target[i,1].item(), mu[i,0].item(), mu[i,1].item(), dist_pred[i,0].item(), dist_pred[i,1].item())
    
    return area, coverage

def avg_cov_ellipsoid(q, target, mu, sd):
    """
    Efficiently compute the average ellipsoid coverage.
    
    Parameters:
        q (float): Threshold for ellipsoid inclusion.
        target (Tensor): Tensor of shape [N, D] — target points.
        mu (Tensor): Tensor of shape [N, D] — mean of each ellipsoid.
        sd (Tensor): Tensor of shape [N, D] — diagonal std for each ellipsoid.
    
    Returns:
        float: Percentage of points within the ellipsoid.
    """
    # Compute squared Mahalanobis distance for all points in batch
    dists = torch.sum(((target - mu) ** 2) / (sd ** 2), dim=1)  # Shape: [N]
    
    # Check how many distances are within the threshold q
    within = (dists <= q).float()
    
    # Compute the percentage
    return within.mean().item() * 100


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

def avg_cov(dist_true, dist_pred, target):
    in_the_range = torch.sum(dist_true <= dist_pred).item()
    coverage = in_the_range / len(target) * 100
    return coverage

def ellipsoid_volumes(radii_batch):
    """
    Calculate volumes of a batch of k-dimensional ellipsoids.
    
    Parameters:
        radii_batch (Tensor): Shape [N, D], where each row is a radii vector.
    
    Returns:
        Tensor: Shape [N], volume for each ellipsoid.
    """
    N, D = radii_batch.shape
    volume_unit_ball = math.pi ** (D / 2) / math.gamma(D / 2 + 1)
    prod_radii = torch.prod(radii_batch, dim=1)  # Shape: [N]
    return volume_unit_ball * prod_radii

def avg_ellipsoid_volume(dist_pred):
    """
    Compute average ellipsoid volume for a batch of radii vectors.
    
    Parameters:
        dist_pred (Tensor): Shape [N, D] — batch of radii vectors.
    
    Returns:
        float: Average volume.
    """
    volumes = ellipsoid_volumes(dist_pred)
    return volumes.mean().item()

class DataToVisualize:
    def __init__(self, x, mu, sd, target):
        """
        Initialize the DataToVisualize object with input data, predictions, and standard deviations.
        
        Args:
            x (Tensor): Input data tensor.
            mu (Tensor): Predicted mean tensor.
            sd (Tensor): Standard deviation tensor.
        """
        self.x = x
        self.mu = mu
        self.sd = sd
        self.target = target


def get_arrays(data_loader, model, device):
    y_p_s = []
    vars_s = []
    logvars_s = []
    targets_s = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)

            y_p, logvar, var_bayesian = model(data, dropout=True, mc_dropout=True, test=True)


            y_p_s.append(y_p.detach())
            vars_s.append(var_bayesian.detach())
            logvars_s.append(logvar.detach())
            targets_s.append(target.detach())
            if batch_idx ==0:
                data_to_visualize = data.cpu().numpy()




    targets = torch.cat(targets_s).cpu()
    mu = torch.cat(y_p_s, dim=1).clamp(0, 1).permute(1,0,2)
    mu = mu.mean(dim=1) .cpu()    
    var = torch.cat(vars_s, dim=0).cpu()
    logvar = torch.cat(logvars_s, dim=1).permute(1,0,2)
    logvar = logvar.mean(dim=1).cpu()
    
    visaul_data_dim = data_to_visualize.shape[0]
    mu_to_visualize = mu[:visaul_data_dim]
    logvar_to_visualize = logvar[:visaul_data_dim]
    target_to_visualize = targets[:visaul_data_dim]
    save_visual_data = DataToVisualize(data_to_visualize, mu_to_visualize, logvar_to_visualize.exp().sqrt(), target_to_visualize)
                    
    return mu, var, logvar, targets , save_visual_data
    


def shuffle_arrays(calib_arrays, test_arrays):
    """
    Shuffles calibration and test arrays together, maintaining correspondence across arrays.

    Args:
        calib_arrays (list of tensors): List of calibration arrays to shuffle.
        test_arrays (list of tensors): List of test arrays to shuffle.
        seed (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        tuple: Shuffled calibration arrays, shuffled test arrays.
    """
    # if seed is not None:
    #     np.random.seed(seed)

    # Combine calib and test arrays
    combined_arrays = [torch.cat([calib, test], dim=0) for calib, test in zip(calib_arrays, test_arrays)]
    
    # Generate shuffle indices
    total_length = combined_arrays[0].shape[0]
    shuffle_indices = np.random.permutation(total_length)

    # Apply shuffle indices
    shuffled_arrays = [arr[shuffle_indices] for arr in combined_arrays]

    # Split back into calib and test arrays
    split_index = len(calib_arrays[0])
    calib_shuffled = [arr[:split_index] for arr in shuffled_arrays]
    test_shuffled = [arr[split_index:] for arr in shuffled_arrays]

    return calib_shuffled, test_shuffled
   
    
    

def main():
    print("Current PID:", os.getpid())
    mix_indices = True
    save_params = False
    load_params = False
    save_test = True
    load_test = False
    calc_mean = False
    eval_test_set( save_params=save_params, mix_indices=mix_indices, load_params=load_params, calc_mean=calc_mean, save_test=save_test, load_test=load_test)

def eval_test_set(save_params=False, load_params=False, mix_indices=True, calc_mean=False, save_test=False, load_test=False):
    base_model = 'efficientnetb4'
    models_dir = '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots'
    assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
    device = torch.device("cuda:0")
    dataset = 'oct'
    iters = 20
    level = 1
    alpha = 0.05
    load_preds = False
    save_visual = False
    
    print(f'alpha: {alpha}, level: {level}, base_model: {base_model}, mix_indices: {mix_indices}, save_params: {save_params}, load_params: {load_params}, calc_mean: {calc_mean}, save_test: {save_test}, load_test: {load_test}')
    

    
    batch_size = 64


    

    
    results_dir = "/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results/predictions/dims"

    # save test arrays
    if not load_preds:
        

        # TODO: load checkpoint 
        # checkpoint_path = glob(f"/home/dsi/frenkel2/regression_calibration/models/{base_model}_gaussian_endovis_199_new.pth.tar")[0]
        # checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\{base_model}_gaussian_endovis_199_new.pth.tar")[0]
        # TODO fix it 
        if dataset == 'brain':
            checkpoint = torch.load(f'{models_dir}/{base_model}_gaussian_{dataset}_best.pth.tar', map_location=device)
        elif dataset == 'oct':
            out_channels = 6
            checkpoint = torch.load(f'{models_dir}/{base_model}_gaussian_oct_best_dims.pth.tar', map_location=device)
        else:
            out_channels = 2
            checkpoint = torch.load(f'{models_dir}/{base_model}_gaussian_lumbar_L{level}_best_dims.pth.tar', map_location=device)
        model = BreastPathQModel(base_model, out_channels=out_channels).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"epoch: {checkpoint['epoch']}")
        model.eval()
        if dataset =='brain':
            data_set_valid_original = BrainDatasetVal(model=base_model)
            data_set_test_original = BrainDatasetTest(model=base_model)
        elif dataset == 'oct':
            resize_to = (256, 256)

            data_set_valid_original = OCTDataset(group='valid', augment=False, resize_to=resize_to)
            data_set_test_original = OCTDataset(group='test', augment=False, resize_to=resize_to)
        else:
            data_set_valid_original = LumbarDataset(level=level, mode='val', augment=False, scale=0.5)
            data_set_test_original = LumbarDataset(level=level, mode='test', augment=False, scale=0.5)
        
        assert len(data_set_valid_original) > 0
        assert len(data_set_test_original) > 0
        print(len(data_set_valid_original))
        print(len(data_set_test_original))
        
        calib_loader = torch.utils.data.DataLoader(data_set_valid_original, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(data_set_test_original, batch_size=batch_size, shuffle=False)
        y_p_test_original, vars_test_original, logvars_test_original, targets_test_original, test_visual = get_arrays(test_loader, model, device)
        y_p_calib_original, vars_calib_original, logvars_calib_original, targets_calib_original, valid_visual = get_arrays(calib_loader, model, device)

        if save_visual:
            torch.save(
                {
                'x' : test_visual.x,
                'mu': test_visual.mu,
                'sd': test_visual.sd,
                'target': test_visual.target}, f'{results_dir}/{dataset}_dataset_model_{base_model}_level{level}_test_visual.pth.tar')
        
        np.save(f'{results_dir}/{dataset}_dataset_model_{base_model}_level{level}_y_p_test_original.npy', y_p_test_original.cpu().numpy())
        np.save(f'{results_dir}/{dataset}_dataset_model_{base_model}_level{level}_logvars_test_original.npy', logvars_test_original.cpu().numpy())
        np.save(f'{results_dir}/{dataset}_dataset_model_{base_model}_level{level}_targets_test_original.npy', targets_test_original.cpu().numpy())  
        np.save(f'{results_dir}/{dataset}_dataset_model_{base_model}_level{level}_y_p_calib_original.npy', y_p_calib_original.cpu().numpy()) 
        np.save(f'{results_dir}/{dataset}_dataset_model_{base_model}_level{level}_logvars_calib_original.npy', logvars_calib_original.cpu().numpy())
        np.save(f'{results_dir}/{dataset}_dataset_model_{base_model}_level{level}_targets_calib_original.npy', targets_calib_original.cpu().numpy())
    else:
        y_p_calib_original = torch.from_numpy(np.load(f'{results_dir}/{dataset}_dataset_model_{base_model}_level{level}_y_p_calib_original.npy'))
        logvars_calib_original = torch.from_numpy(np.load(f'{results_dir}/{dataset}_dataset_model_{base_model}_level{level}_logvars_calib_original.npy'))
        vars_calib_original = logvars_calib_original.exp()
        targets_calib_original = torch.from_numpy(np.load(f'{results_dir}/{dataset}_dataset_model_{base_model}_level{level}_targets_calib_original.npy'))
        y_p_test_original = torch.from_numpy(np.load(f'{results_dir}/{dataset}_dataset_model_{base_model}_level{level}_y_p_test_original.npy'))
        logvars_test_original = torch.from_numpy(np.load(f'{results_dir}/{dataset}_dataset_model_{base_model}_level{level}_logvars_test_original.npy'))
        vars_test_original = logvars_test_original.exp()
        targets_test_original = torch.from_numpy(np.load(f'{results_dir}/{dataset}_dataset_model_{base_model}_level{level}_targets_test_original.npy'))
        
    
    # print(f"y_p_test: {list(y_p_test_original)}")
    # print(f"logvars_test: {list(logvars_test_original)}")
    
    # print(f"y_p_calib: {list(y_p_calib_original)}")
    # print(f"logvars_calib: {list(logvars_calib_original)}")
    
    # Calibration and test arrays (from your original code)
    calib_arrays = [
        y_p_calib_original, 
        vars_calib_original, 
        logvars_calib_original, 
        targets_calib_original, 
    ]

    test_arrays = [
        y_p_test_original, 
        vars_test_original, 
        logvars_test_original, 
        targets_test_original, 
    ]

    q_all = []
    avg_len_all = []
    avg_cov_all = []
    q_all_gc = []
    avg_len_all_gc = []
    avg_cov_all_gc = []
    q_all_bonf = []
    avg_len_all_bonf = []
    avg_cov_all_bonf = []
    q_all_max_rank = []
    avg_len_all_max_rank = []
    avg_cov_all_max_rank = []
    # validation results
    
    avg_len_valid_all = []
    avg_cov_valid_all = []
    avg_len_valid_all_gc = []
    avg_cov_valid_all_gc = []
    avg_len_valid_all_bonf = []
    avg_cov_valid_all_bonf = []
    avg_len_valid_all_max_rank = []
    avg_cov_valid_all_max_rank = []
    


    for j in range(iters):
        print(f'Iter: {j}')
        y_p_calib = []
        vars_calib = []
        logvars_calib = []
        targets_calib = []
        
        if mix_indices:
            calib_shuffled, test_shuffled = shuffle_arrays(calib_arrays, test_arrays)
            y_p_calib, vars_calib, logvars_calib, targets_calib = calib_shuffled
            y_p_test, vars_test, logvars_test, targets_test = test_shuffled
        else:
            y_p_calib, vars_calib, logvars_calib, targets_calib = y_p_calib_original, vars_calib_original, logvars_calib_original, targets_calib_original, 
            y_p_test, vars_test, logvars_test, targets_test = y_p_test_original, vars_test_original, logvars_test_original, targets_test_original
        
                    
        # validation set   
        y_p_calib = y_p_calib.clamp(0, 1)
        mu_calib = y_p_calib
        var_calib = vars_calib
        logvars_calib = logvars_calib
        logvar_calib = logvars_calib
        var_calib  = logvars_calib.exp()
        sd_calib = var_calib.sqrt()
        target_calib = targets_calib




        
        # test set
                                 
        y_p_test = y_p_test.clamp(0, 1)
        mu_test = y_p_test
        logvars_test = logvars_test
        logvar_test = logvars_test
        var_test = logvar_test.exp()
        sd_test = var_test.sqrt()
        target_test = targets_test
            
        q = calc_optimal_q(target_calib, mu_calib, sd_calib, alpha)
        
        valid_length, valid_coverage = calc_stats(q, target_calib, mu_calib, sd_calib)
        test_length, test_coverage = calc_stats(q, target_test, mu_test, sd_test)

            
        q_gc = calc_optimal_q(target_calib, mu_calib, sd_calib, alpha, gc=True)
                     
        valid_length_gc, valid_coverage_gc = calc_stats(q_gc, target_calib, mu_calib, sd_calib)
        test_length_gc, test_coverage_gc = calc_stats(q_gc, target_test, mu_test, sd_test)
        
        
        q_bonf = calc_bonferroni_q(target_calib, mu_calib, sd_calib, alpha)
        valid_length_bonf, valid_coverage_bonf = compute_coverage_size_bonf(target_calib, mu_calib, sd_calib, q_bonf)
        test_length_bonf, test_coverage_bonf = compute_coverage_size_bonf(target_test, mu_test, sd_test, q_bonf)

        q_max_rank = calc_optimal_q_max_rank(target_calib, mu_calib, sd_calib, alpha)
        valid_length_max_rank, valid_coverage_max_rank = compute_coverage_size_bonf(target_calib, mu_calib, sd_calib, q_max_rank)
        test_length_max_rank, test_coverage_max_rank = compute_coverage_size_bonf( target_test, mu_test, sd_test, q_max_rank)
          
        print(f'q: {q}, q_gc: {q_gc}, q_bonf: {q_bonf}')
        print(f'valid_length: {valid_length}, valid_coverage: {valid_coverage}')
        print(f'test_length: {test_length}, test_coverage: {test_coverage}')
        print(f'valid_length_gc: {valid_length_gc}, valid_coverage_gc: {valid_coverage_gc}')
        print(f'test_length_gc: {test_length_gc}, test_coverage_gc: {test_coverage_gc}')
        print(f'valid_length_bonf: {valid_length_bonf}, valid_coverage_bonf: {valid_coverage_bonf}')
        print(f'test_length_bonf: {test_length_bonf}, test_coverage_bonf: {test_coverage_bonf}')
        print(f'valid_length_max_rank: {valid_length_max_rank}, valid_coverage_max_rank: {valid_coverage_max_rank}')
        print(f'test_length_max_rank: {test_length_max_rank}, test_coverage_max_rank: {test_coverage_max_rank}')
        
            

        q_all.append(get_float(q))
        avg_len_all.append(get_float(test_length))
        avg_cov_all.append(get_float(test_coverage))
        
        q_all_gc.append(get_float(q_gc))
        avg_len_all_gc.append(get_float(test_length_gc))
        avg_cov_all_gc.append(get_float(test_coverage_gc))
        
        avg_len_valid_all.append(get_float(valid_length))
        avg_cov_valid_all.append(get_float(valid_coverage))
        avg_len_valid_all_gc.append(get_float(valid_length_gc))
        avg_cov_valid_all_gc.append(get_float(valid_coverage_gc))
        
        q_all_bonf.append(q_bonf)
        avg_cov_all_bonf.append(get_float(test_coverage_bonf))
        avg_len_all_bonf.append(get_float(test_length_bonf))
        avg_len_valid_all_bonf.append(get_float(valid_length_bonf))
        avg_cov_valid_all_bonf.append(get_float(valid_coverage_bonf))
        
        
        q_all_max_rank.append(q_max_rank)
        avg_len_all_max_rank.append(get_float(test_length_max_rank))
        avg_cov_all_max_rank.append(get_float(test_coverage_max_rank))
        avg_len_valid_all_max_rank.append(get_float(valid_length_max_rank))
        avg_cov_valid_all_max_rank.append(get_float(valid_coverage_max_rank))
        
        
    print(f"q cp: {q_all}")
    print(f"q gc: {q_all_gc}")
    print(f"avg_len cp: {avg_len_all}")
    print(f"avg_len gc: {avg_len_all_gc}")
    print(f"avg_cov cp: {avg_cov_all}")
    print(f"avg_cov gc: {avg_cov_all_gc}")
    print(f"avg_len bonf: {avg_len_all_bonf}")
    print(f"avg_cov bonf: {avg_cov_all_bonf}")

    # Define the output file path
    output_dir= '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results/dims'
    output_file = f"{dataset}_dataset_model_{base_model}_alpha_{alpha}_level_{level}_iterations_{iters}.txt"

    # Open the file in append mode
    with open(f'{output_dir}/{output_file}', "w") as f:
        # Print and save CP metrics
        print(f'q CP mean: {statistics.mean(q_all)}, q CP std: {statistics.stdev(q_all)}')
        f.write(f'q CP mean: {statistics.mean(q_all)}, q CP std: {statistics.stdev(q_all)}\n')
        
        print(f'avg_size CP mean: {statistics.mean(avg_len_all)}, avg_size CP std: {statistics.stdev(avg_len_all)}')
        f.write(f'avg_size CP mean: {statistics.mean(avg_len_all)}, avg_size CP std: {statistics.stdev(avg_len_all)}\n')
        
        print(f'avg_cov CP mean: {statistics.mean(avg_cov_all)}, avg_cov CP std: {statistics.stdev(avg_cov_all)}')
        f.write(f'avg_cov CP mean: {statistics.mean(avg_cov_all)}, avg_cov CP std: {statistics.stdev(avg_cov_all)}\n')
        
        # Print and save GC metrics
        print(f'q GC mean: {statistics.mean(q_all_gc)}, q GC std: {statistics.stdev(q_all_gc)}')
        f.write(f'q GC mean: {statistics.mean(q_all_gc)}, q GC std: {statistics.stdev(q_all_gc)}\n')
        
        print(f'avg_size GC mean: {statistics.mean(avg_len_all_gc)}, avg_size GC std: {statistics.stdev(avg_len_all_gc)}')
        f.write(f'avg_szie GC mean: {statistics.mean(avg_len_all_gc)}, avg_size GC std: {statistics.stdev(avg_len_all_gc)}\n')
        
        print(f'avg_cov GC mean: {statistics.mean(avg_cov_all_gc)}, avg_cov GC std: {statistics.stdev(avg_cov_all_gc)}')
        f.write(f'avg_cov GC mean: {statistics.mean(avg_cov_all_gc)}, avg_cov GC std: {statistics.stdev(avg_cov_all_gc)}\n')
        
        # Print and save Bonferroni metrics
        print(f'q Bonf mean: {np.array(q_all_bonf).mean(axis=0).tolist()}, q Bonf std: {np.array(q_all_bonf).std(axis=0).tolist()}')
        f.write(f'q Bonf mean: {np.array(q_all_bonf).mean(axis=0).tolist()}, q Bonf std: {np.array(q_all_bonf).std(axis=0).tolist()}\n')
        
        print(f'avg_size Bonf mean: {statistics.mean(avg_len_valid_all_bonf)}, avg_size Bonf std: {statistics.stdev(avg_len_valid_all_bonf)}')
        f.write(f'avg_size Bonf mean: {statistics.mean(avg_len_valid_all_bonf)}, avg_size Bonf std: {statistics.stdev(avg_len_valid_all_bonf)}\n')
        print(f'avg_cov Bonf mean: {statistics.mean(avg_cov_all_bonf)}, avg_cov Bonf std: {statistics.stdev(avg_cov_all_bonf)}')
        f.write(f'avg_cov Bonf mean: {statistics.mean(avg_cov_all_bonf)}, avg_cov Bonf std: {statistics.stdev(avg_cov_all_bonf)}\n')
    
        print(f'q max_rank mean: {np.array(q_all_max_rank).mean(axis=0).tolist()}, q max_rank std: {np.array(q_all_max_rank).std(axis=0).tolist()}')
        f.write(f'q max_rank mean: {np.array(q_all_max_rank).mean(axis=0).tolist()}, q max_rank std: {np.array(q_all_max_rank).std(axis=0).tolist()}\n')
        
        print(f'avg_size max_rank mean: {statistics.mean(avg_len_valid_all_max_rank)}, avg_size max_rank std: {statistics.stdev(avg_len_valid_all_max_rank)}')
        f.write(f'avg_size max_rank mean: {statistics.mean(avg_len_valid_all_max_rank)}, avg_size max_rank std: {statistics.stdev(avg_len_valid_all_max_rank)}\n')
        
        print(f'avg_cov max_rank mean: {statistics.mean(avg_cov_valid_all_max_rank)}, avg_cov max_rank std: {statistics.stdev(avg_cov_valid_all_max_rank)}')
        f.write(f'avg_cov max_rank mean: {statistics.mean(avg_cov_valid_all_max_rank)}, avg_cov max_rank std: {statistics.stdev(avg_cov_valid_all_max_rank)}\n')
        
        # save validation results
        f.write(f"Validation results:\n")
        print(f"avg_size validation mean: {statistics.mean(avg_len_valid_all)}, avg_size validation std: {statistics.stdev(avg_len_valid_all)}")
        f.write(f"avg_size validation mean: {statistics.mean(avg_len_valid_all)}, avg_size validation std: {statistics.stdev(avg_len_valid_all)}\n")
        
        print(f"avg_cov validation mean: {statistics.mean(avg_cov_valid_all)}, avg_cov validation std: {statistics.stdev(avg_cov_valid_all)}")
        f.write(f"avg_cov validation mean: {statistics.mean(avg_cov_valid_all)}, avg_cov validation std: {statistics.stdev(avg_cov_valid_all)}\n")
        
        print(f"avg_size validation mean GC: {statistics.mean(avg_len_valid_all_gc)}, avg_size validation std GC: {statistics.stdev(avg_len_valid_all_gc)}")
        f.write(f"avg_size validation mean GC: {statistics.mean(avg_len_valid_all_gc)}, avg_size validation std GC: {statistics.stdev(avg_len_valid_all_gc)}\n")
        
        print(f"avg_cov validation mean GC: {statistics.mean(avg_cov_valid_all_gc)}, avg_cov validation std GC: {statistics.stdev(avg_cov_valid_all_gc)}")
        f.write(f"avg_cov validation mean GC: {statistics.mean(avg_cov_valid_all_gc)}, avg_cov validation std GC: {statistics.stdev(avg_cov_valid_all_gc)}\n")        
        
        print(f"avg_size validation mean Bonf: {statistics.mean(avg_len_valid_all_bonf)}, avg_size validation std Bonf: {statistics.stdev(avg_len_valid_all_bonf)}")
        f.write(f"avg_size validation mean Bonf: {statistics.mean(avg_len_valid_all_bonf)}, avg_size validation std Bonf: {statistics.stdev(avg_len_valid_all_bonf)}\n")
        
        print(f"avg_cov validation mean Bonf: {statistics.mean(avg_cov_valid_all_bonf)}, avg_cov validation std Bonf: {statistics.stdev(avg_cov_valid_all_bonf)}")
        f.write(f"avg_cov validation mean Bonf: {statistics.mean(avg_cov_valid_all_bonf)}, avg_cov validation std Bonf: {statistics.stdev(avg_cov_valid_all_bonf)}\n")
        
        print(f"avg_size validation mean max_rank: {statistics.mean(avg_len_valid_all_max_rank)}, avg_size validation std max_rank: {statistics.stdev(avg_len_valid_all_max_rank)}")
        f.write(f"avg_size validation mean max_rank: {statistics.mean(avg_len_valid_all_max_rank)}, avg_size validation std max_rank: {statistics.stdev(avg_len_valid_all_max_rank)}\n")
        print(f"avg_cov validation mean max_rank: {statistics.mean(avg_cov_valid_all_max_rank)}, avg_cov validation std max_rank: {statistics.stdev(avg_cov_valid_all_max_rank)}")
        f.write(f"avg_cov validation mean max_rank: {statistics.mean(avg_cov_valid_all_max_rank)}, avg_cov validation std max_rank: {statistics.stdev(avg_cov_valid_all_max_rank)}\n")             
    
        # Print and save additional info
        print(f"{dataset}, {base_model}, {alpha}, {level}")
        f.write(f"{dataset}, {base_model}, {alpha}, {level}\n")
    
  
def get_float(x):
    try:
        return x.item()
    except:
        return x
  
    


    
    
if __name__ == '__main__':
    main()