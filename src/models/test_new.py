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
from data_generator_lumbar import LumbarDataset
from data_generator_boneage import BoneAgeDataset
from models import BreastPathQModel, DistancePredictor, DistancePredictorOneOutput, DistNewModel
from glob import glob
import statistics
import math
import load_trained_models
import numpy as np
import torch
import random
from scipy.stats import norm



seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def calc_opt_q_new_method(target_calib, mu_calib, poistive_dist, negative_dist, alpha, addtvie):
    if addtvie:
        # c_min = mu_calib - negative_dist - target_calib
        # c_max =  target_calib - mu_calib - poistive_dist
        c_min = negative_dist - target_calib
        c_max =  target_calib - poistive_dist
        c = torch.max(c_min, c_max)
    else:
        # c_min  = (mu_calib - target_calib) / negative_dist
        # c_max  = (target_calib - mu_calib) / poistive_dist
        c_min = negative_dist/target_calib
        c_max = target_calib/poistive_dist
        c = torch.max(c_min, c_max)
    c_sorted, _ = torch.sort(c, dim=0)
    q_index = math.ceil((len(c_sorted)) * (1 - alpha))
    q = c_sorted[q_index].item()
    return q

def calc_opt_q_new_method_my_mu(target_calib, mu_pos, mu_neg, poistive_dist, negative_dist, alpha):
    c_min  = (mu_neg - target_calib) / negative_dist
    c_max  = (target_calib - mu_pos) / poistive_dist
    c = torch.max(c_min, c_max)
    c_sorted, _ = torch.sort(c, dim=0)
    q_index = math.ceil((len(c_sorted)) * (1 - alpha))
    q = c_sorted[q_index].item()
    return q


def calc_coverage_add(mu, target, poistive_dist, negative_dist, q):
    # y_upper = mu + poistive_dist + q
    # y_lower = mu - negative_dist - q
    y_upper = poistive_dist + q
    y_lower = negative_dist - q
    return calc_coverage(y_lower, y_upper, target)

def calc_coverage_div(target, mu, poistive_dist, negative_dist, q):
    # y_upper = mu + poistive_dist * q
    # y_lower = mu - negative_dist * q
    y_upper = poistive_dist * q
    y_lower = negative_dist /q
    return calc_coverage(y_lower, y_upper, target)


def calc_stats_new_method(target, mu, poistive_dist, negative_dist, q, div = False):
    if div:
        # lower = mu - negative_dist * q
        # upper = mu + poistive_dist * q
        upper = poistive_dist * q
        lower = negative_dist / q
    else:
        # lower = mu - negative_dist - q
        # upper = mu + poistive_dist + q
        lower = negative_dist - q
        upper = poistive_dist + q
    length = calc_length(lower, upper)
    coverage = calc_coverage(lower, upper, target)
    return length, coverage

def calc_stats_new_method_my_mu(target, mu_pos, mu_neg, poistive_dist, negative_dist, q):
    lower = mu_neg - negative_dist * q
    upper = mu_pos + poistive_dist * q
    length = calc_length(lower, upper)
    coverage = calc_coverage(lower, upper, target)
    return length, coverage


def calc_stats(q, target, mu, sd):
    lower = mu - q * sd
    upper = mu + q * sd
    length = calc_length(lower, upper)
    coverage = calc_coverage(lower, upper, target)
    return length, coverage

def  calc_coverage(lower, upper, target):
    coverage = (lower <= target) & (target <= upper)
    return coverage.float().mean().item()

def calc_length(lower, upper):
    lower = torch.clamp(lower, min=0, max=1) 
    upper = torch.clamp(upper, min=0, max=1)  
    return torch.mean(abs(upper - lower)).item()
    
    

# CP
def calc_optimal_q(target_calib, mu_calib, sd_calib, alpha, gc=False):

    s_t = torch.abs(target_calib-mu_calib) / sd_calib
    if gc:
        S = (s_t).mean().sqrt()
        multiplier = norm.ppf(1 - alpha)
        q = multiplier * S.item()
        # if alpha == 0.1:
        #     q = 1.64485 * S.item()
        # elif alpha == 0.05:
        #     q = 1.95996 * S.item()
        # else:
        #     print("Choose another value of alpha!! (0.1 / 0.05)")
    else:
        s_t_sorted, _ = torch.sort(s_t, dim=0)
        q_index = math.ceil((len(s_t_sorted)) * (1 - alpha))
        q = s_t_sorted[q_index].item()   
    return q


def get_saved_dir(results_dir, dataset, base_model, dist_model, loss,group,  level = None, lambda_param = 1, scale_factor = 1):
    cur_dir = results_dir
    os.makedirs(cur_dir, exist_ok=True)
    cur_dir = f"{cur_dir}/{dataset}"
    os.makedirs(cur_dir, exist_ok=True)
    if level != None:
        cur_dir = f"{cur_dir}/{level}"
        os.makedirs(cur_dir, exist_ok=True)
    cur_dir = f"{cur_dir}/{loss}"
    os.makedirs(cur_dir, exist_ok=True)
    cur_dir = f"{cur_dir}/{base_model}"
    os.makedirs(cur_dir, exist_ok=True)
    cur_dir = f"{cur_dir}/{dist_model}"
    os.makedirs(cur_dir, exist_ok=True)
    cur_dir = f"{cur_dir}/{group}"
    os.makedirs(cur_dir, exist_ok=True)
    cur_dir = f"{cur_dir}/{lambda_param}"
    os.makedirs(cur_dir, exist_ok=True)
    cur_dir =f"{cur_dir}/{int(scale_factor)}"
    os.makedirs(cur_dir, exist_ok=True)
    return cur_dir

    

def save_arrays(results_dir, dataset, base_model, dist_model, loss, group, level, lambda_param, scale_factor, y, mu, logvar, positive_distance, negative_distance, right_dist_calib_original, left_dist_calib_original):
    saved_dir = get_saved_dir(results_dir=results_dir, dataset=dataset, base_model=base_model, dist_model=dist_model, loss = loss, group = group, level=level, lambda_param = lambda_param, scale_factor = scale_factor)
    np.save(f'{saved_dir}/y.npy', y.cpu().numpy())
    np.save(f'{saved_dir}/mu.npy', mu.cpu().numpy())
    np.save(f'{saved_dir}/logvar.npy', logvar.cpu().numpy())  
    np.save(f'{saved_dir}/positive_distance.npy', positive_distance.cpu().numpy()) 
    np.save(f'{saved_dir}/negative_distance.npy', negative_distance)
    np.save(f'{saved_dir}/right_distance.npy', right_dist_calib_original.cpu().numpy())
    np.save(f'{saved_dir}/left_distance.npy', left_dist_calib_original.cpu().numpy())

def load_arrays(results_dir, dataset, base_model, dist_model, loss, group, level, lambda_param, scale_factor):
    saved_dir = get_saved_dir(results_dir=results_dir, dataset=dataset, base_model=base_model, dist_model=dist_model, loss = loss, group = group, level=level, lambda_param = lambda_param, scale_factor = scale_factor)
    y = np.load(f'{saved_dir}/y.npy')
    mu = np.load(f'{saved_dir}/mu.npy')
    logvar = np.load(f'{saved_dir}/logvar.npy')  
    pos_dist = np.load(f'{saved_dir}/positive_distance.npy') 
    neg_dist = np.load(f'{saved_dir}/negative_distance.npy')
    right_dist = np.load(f'{saved_dir}/right_distance.npy')
    left_dist = np.load(f'{saved_dir}/left_distance.npy')
    return torch.from_numpy(y), torch.from_numpy(mu), torch.from_numpy(logvar), torch.from_numpy(pos_dist), torch.from_numpy(neg_dist), torch.from_numpy(right_dist), torch.from_numpy(left_dist)
    
def modify_predicted_distances(predicted_distances, probs):
    first_dim_results = predicted_distances * probs
    second_dim_results = predicted_distances * (1 - probs)
    return torch.stack([first_dim_results, second_dim_results], dim=1).squeeze(-1)

def get_arrays(data_loader, model, dist_model, device, one_output = False, scale_factor= 1.0):
    r_limit_s = []
    vars_s = []
    l_limit_s = []
    targets_s = []
    positive_dist_s = []
    negative_dist_s = []
    right_dist_s = []
    left_dist_s = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)

            r_limit, l_limit = model(data, dropout=False, mc_dropout=False, test=False)

            r_limit_s.append(r_limit.detach())
            vars_s.append(r_limit.detach())
            l_limit_s.append(l_limit.detach())
            targets_s.append(target.detach()) 
                        
            if dist_model is not None:
                # distances = dist_model(data).detach()
                distances = dist_model(data)
                if one_output:
                    positive_dist_s.append(distances.squeeze(-1))
                    negative_dist_s.append(distances.squeeze(-1))
                else:
                    positive_dist_s.append(r_limit)
                    negative_dist_s.append(l_limit)
                    right_dist_s.append(distances[0].squeeze(-1).detach())
                    left_dist_s.append(distances[1].squeeze(-1).detach())


            
                    
    return torch.cat(r_limit_s).cpu(), torch.cat(vars_s).cpu(), torch.cat(l_limit_s).cpu(), torch.cat(targets_s).cpu(), torch.cat(positive_dist_s).cpu() / float(scale_factor), torch.cat(negative_dist_s).cpu() / float(scale_factor)   , torch.cat(right_dist_s).cpu(), torch.cat(left_dist_s).cpu()  
    


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
   

def get_dir_results(one_output = False):
    resutls_dir_path = '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results_new'
    if one_output:
        resutls_dir_path = f"{resutls_dir_path}/one_output"
    os.makedirs(resutls_dir_path, exist_ok=True)
    return resutls_dir_path
    
def normalize_dist(dist, logvar):
    return dist * logvar.squeeze(-1).exp().sqrt()

def print_first_10_elements(**arrays):
    for name, array in arrays.items():
        print(f"{name}: {array[:10]}")
        
        
def sort_two_tensors(tensor1, tensor2):
    # Stack tensors along a new dimension
    stacked = torch.stack([tensor1, tensor2], dim=0)

    # Sort along that dimension
    sorted_values, _ = torch.sort(stacked, dim=0)

    # Extract max, middle, and min tensors
    min_values = sorted_values[0]  # Smallest values
    max_values = sorted_values[1]  # Largest values

    return max_values, min_values

def load_dist_model(base_model, device, level, upper=False):
    dist_model = BreastPathQModel(base_model, out_channels=1).to(device) 
    if upper:
        side= 'upper'
    else:
        side = 'lower'
    checkpoint = torch.load(f"/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots_new/efficientnetb4_lumbar_L{level}_snapshot_dist_efficientnetb4_lambda_1_scale_factor1_{side}_new.pth.tar", map_location=device)
    dist_model.load_state_dict(checkpoint['state_dict'])
    print(f"epoch: {checkpoint['epoch']}")
    dist_model.eval()
    return dist_model
        
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
    base_model = 'densenet201'
    base_model_dist = 'resnet50'
    assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
    device = torch.device("cuda:2")
    dataset = 'lumbar'
    loss = 'gaussian'
    pred_x = False
    pred_y = False
    one_output = False
    load_results = True
    normalize = False
    scale_factor = 1.0
    lambda_param = 1
    iters = 20
    level = 4
    alpha = 0.05

    
    
    print(f'alpha: {alpha}, level: {level}, base_model: {base_model}, mix_indices: {mix_indices}, save_params: {save_params}, load_params: {load_params}, calc_mean: {calc_mean}, save_test: {save_test}, load_test: {load_test}')
    
    
    cur_level = level
    if dataset == 'boneage':
        cur_level = None
    batch_size = 64
    results_dir = '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results_new/predictions'
    if not load_results:
        if dataset == 'lumbar':
            # model = load_trained_models.get_model_lumbar(base_model, level, None, device, loss=loss, pred_x=pred_x, pred_y=pred_x)
            # dist_model = load_trained_models.get_model_lumbar(base_model_dist, level, base_model, device, lambda_param=lambda_param, one_out=one_output, loss=loss, pred_x=pred_x, pred_y=pred_x, normalize=normalize)
            # dist_model_upper = load_dist_model(base_model, device, level, upper=True)
            # dist_model_lower = load_dist_model(base_model, device, level, upper=False)   
            model = DistNewModel(base_model, out_channels=1).to(device) 
            checkpoint = torch.load(f"/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots_new/{base_model}_lumbar_L{level}_snapshot_dist_{base_model}_lambda_1_scale_factor1_best.pth.tar", map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"epoch: {checkpoint['epoch']}")
            model.eval()  
                 
            dist_model = DistNewModel(base_model, out_channels=1).to(device) 
            checkpoint = torch.load(f"/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots_new/{base_model}_lumbar_L{level}_snapshot_dist_{base_model}_lambda_1_dist_scale_factor1_best.pth.tar", map_location=device)
            dist_model.load_state_dict(checkpoint['state_dict'])
            print(f"epoch: {checkpoint['epoch']}")
            dist_model.eval()    
            data_set_valid_original = LumbarDataset(level=level, mode='val', augment=False, pred_x=pred_x, pred_y=pred_x, scale=0.5)
            data_set_test_original = LumbarDataset(level=level, mode='test', augment=False, pred_x=pred_x, pred_y=pred_x, scale=0.5)
        elif dataset == 'boneage':
            resize_to = (256, 256)
            data_set_valid_original = BoneAgeDataset(group='valid', augment=False, resize_to=resize_to)
            data_set_test_original = BoneAgeDataset(group='test', augment=False, resize_to=resize_to)
            model = load_trained_models.get_model_boneage(base_model, None, device, loss=loss)
            dist_model_upper = load_trained_models.get_model_boneage(base_model_dist, base_model, device, lambda_param=lambda_param, one_out=one_output, loss=loss)
        else:
            assert False
    
    
        assert len(data_set_valid_original) > 0
        assert len(data_set_test_original) > 0
        print(len(data_set_valid_original))
        print(len(data_set_test_original))
            
        calib_loader = torch.utils.data.DataLoader(data_set_valid_original, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(data_set_test_original, batch_size=batch_size, shuffle=False)
        y_p_calib_original, vars_calib_original, logvars_calib_original, targets_calib_original, positive_dist_calib_original, negative_dist_calib_original, right_dist_calib_original, left_dist_calib_original = get_arrays(calib_loader, model, dist_model, device, one_output=one_output, scale_factor= scale_factor)
        y_p_test_original, vars_test_original, logvars_test_original, targets_test_original, positive_dist_test_original, negative_dist_test_original, right_dist_test_original, left_dist_test_original = get_arrays(test_loader, model, dist_model, device, one_output=one_output, scale_factor= scale_factor)
        vars_calib_original = logvars_calib_original.exp()
        vars_test_original = logvars_test_original.exp()
        positive_dist_calib_original, negative_dist_calib_original = sort_two_tensors(positive_dist_calib_original, negative_dist_calib_original)
        positive_dist_test_original, negative_dist_test_original = sort_two_tensors(positive_dist_test_original, negative_dist_test_original)
        save_arrays(results_dir = results_dir, dataset = dataset, base_model = base_model, dist_model = base_model_dist, loss = loss, group = 'valid', level = cur_level, lambda_param=lambda_param, scale_factor = scale_factor, y = targets_calib_original, mu=y_p_calib_original, logvar=logvars_calib_original, positive_distance=positive_dist_calib_original, negative_distance= negative_dist_calib_original, right_dist_calib_original=right_dist_calib_original, left_dist_calib_original=left_dist_calib_original)
        save_arrays(results_dir = results_dir, dataset = dataset, base_model = base_model, dist_model = base_model_dist, loss = loss, group = 'test', level = cur_level, lambda_param=lambda_param, scale_factor = scale_factor, y = targets_test_original, mu=y_p_test_original, logvar=logvars_test_original, positive_distance=positive_dist_test_original, negative_distance= negative_dist_test_original, right_dist_calib_original=right_dist_calib_original, left_dist_calib_original=left_dist_calib_original)
    else:
        targets_calib_original, y_p_calib_original,  logvars_calib_original, positive_dist_calib_original, negative_dist_calib_original, right_dist_calib_original, left_dist_calib_original = load_arrays(results_dir = results_dir, dataset = dataset, base_model = base_model, dist_model = base_model_dist, loss = loss, group = 'valid', level = cur_level, lambda_param=lambda_param, scale_factor = scale_factor)
        vars_calib_original = logvars_calib_original.exp()
        targets_test_original, y_p_test_original, logvars_test_original, positive_dist_test_original, negative_dist_test_original, right_dist_test_original, left_dist_test_original = load_arrays(results_dir = results_dir, dataset = dataset, base_model = base_model, dist_model = base_model_dist, loss = loss, group = 'test', level = cur_level, lambda_param = lambda_param, scale_factor = scale_factor,)
        vars_test_original = logvars_test_original.exp()
    
    if normalize:
        positive_dist_calib_original = normalize_dist(positive_dist_calib_original, logvars_calib_original)
        positive_dist_test_original = normalize_dist(positive_dist_test_original, logvars_test_original)
        negative_dist_calib_original = normalize_dist(negative_dist_calib_original, logvars_calib_original)
        negative_dist_test_original = normalize_dist(negative_dist_test_original, logvars_test_original)
    

    # Calibration and test arrays (from your original code)
    calib_arrays = [
        y_p_calib_original, 
        vars_calib_original, 
        logvars_calib_original, 
        targets_calib_original, 
        positive_dist_calib_original, 
        negative_dist_calib_original,
        right_dist_calib_original,
        left_dist_calib_original
    ]

    test_arrays = [
        y_p_test_original, 
        vars_test_original, 
        logvars_test_original, 
        targets_test_original, 
        positive_dist_test_original, 
        negative_dist_test_original,
        right_dist_test_original,
        left_dist_test_original
    ]
    
    print_first_10_elements(
    y_p_calib_original=y_p_calib_original,
    vars_calib_original=vars_calib_original,
    logvars_calib_original=logvars_calib_original,
    targets_calib_original=targets_calib_original
    )
    

    q_all = []
    avg_len_all = []
    avg_cov_all = []
    q_all_gc = []
    avg_len_all_gc = []
    avg_cov_all_gc = []
    
    q_all_new_addtive = []
    avg_len_new_addtive = []
    avg_cov_new_addtive = []
    q_all_new_div = []
    avg_len_new_div = []
    avg_cov_new_div = []
    
    q_mu_all = []
    avg_len_mu_all = []
    avg_cov_mu_all = []


    for j in range(iters):
        print(f'Iter: {j}')
        y_p_calib = []
        vars_calib = []
        logvars_calib = []
        targets_calib = []
        
        if mix_indices:
            calib_shuffled, test_shuffled = shuffle_arrays(calib_arrays, test_arrays)
            y_p_calib, vars_calib, logvars_calib, targets_calib, positive_dist_calib, negative_dist_calib, right_d_calib, left_d_calib = calib_shuffled
            y_p_test, vars_test, logvars_test, targets_test, positive_dist_test, negative_dist_test, right_d_test, left_d_test= test_shuffled
        
                    
        # validation set   
        y_p_calib = y_p_calib.clamp(0, 1).unsqueeze(1)
        mu_calib = y_p_calib.mean(dim=1)
        var_calib = vars_calib
        logvars_calib = logvars_calib
        logvar_calib = logvars_calib.mean(dim=1).unsqueeze(1)
        var_calib  = logvar_calib.exp()
        sd_calib = var_calib.sqrt()
        if dataset== 'boneage':
            target_calib = targets_calib
        else:
            target_calib = targets_calib.unsqueeze(1)
        positive_dist_calib = positive_dist_calib
        negative_dist_calib = negative_dist_calib
        right_d_calib = right_d_calib.unsqueeze(-1)
        left_d_calib = left_d_calib.unsqueeze(-1)
            
        
        # test set
                                 
        y_p_test = y_p_test.clamp(0, 1).unsqueeze(1)
        mu_test = y_p_test.mean(dim=1)
        var_test = vars_test
        logvars_test = logvars_test
        logvar_test = logvars_test.mean(dim=1).unsqueeze(1)
        if dataset== 'boneage':
            target_test = targets_test
        else:
            target_test = targets_test.unsqueeze(1)
        var_test = logvar_test.exp()
        sd_test = var_test.sqrt()
        positive_dist_test = positive_dist_test
        negative_dist_test = negative_dist_test
        right_d_test = right_d_test.unsqueeze(-1)
        left_d_test = left_d_test.unsqueeze(-1)
      
        # CP/GC
        target_calib = target_calib.mean(dim=1, keepdim=True)
        mu_calib = mu_calib.mean(dim=1, keepdim=True)


        # print avg coverage before calib 
        before_cov_val = calc_coverage_add(mu_calib, target_calib, positive_dist_calib, negative_dist_calib, 0)
        before_cov_test = calc_coverage_add(mu_test, target_test, positive_dist_test, negative_dist_test, 0)
        print(f'before_cov_val: {before_cov_val}, before_cov_test: {before_cov_test}')
            
        true_d_plus = torch.clamp(target_calib - mu_calib, min=0) # True d+
        true_d_minus = torch.clamp(mu_calib - targets_calib, min=0) # True d-
        q_add = calc_opt_q_new_method(target_calib, mu_calib, positive_dist_calib, negative_dist_calib, alpha , True)
            
        # cal avg new len and cov valid set
        length_add_calib, coverage_add_calib = calc_stats_new_method(target_calib, mu_calib, positive_dist_calib, negative_dist_calib, q_add, div=False)    
        print(f'q_add: {q_add}, avg_len_single_new_add_val: {length_add_calib}, avg_cov_after_single_new_add_val: {coverage_add_calib}')
            
        # cal avg new len and cov test set
        
        length_add_test, coverage_add_test = calc_stats_new_method(target_test, mu_test, positive_dist_test, negative_dist_test, q_add, div=False)
        print(f'q_add: {q_add}, avg_len_single_new_add_test: {length_add_test}, avg_cov_after_single_new_add_test: {coverage_add_test}')
            
        q_all_new_addtive.append(get_float(q_add))
        avg_len_new_addtive.append(length_add_test)
        avg_cov_new_addtive.append(coverage_add_test)
        
        # prints for ablation 
        lower_dist = torch.abs(q_add+negative_dist_test)
        upper_dist = torch.abs(q_add + positive_dist_test)
        print(f" lower distance: {torch.mean(lower_dist)}, upper distance: {torch.mean(upper_dist)}), ratio: {torch.mean(torch.abs(upper_dist-lower_dist))}, method: addtive")
        
            
        q_div = calc_opt_q_new_method(target_calib, mu_calib, positive_dist_calib, negative_dist_calib, alpha , False)
            
        # cal avg new len and cov valid set
        length_div_calib, coverage_div_calib = calc_stats_new_method(target_calib, mu_calib, positive_dist_calib, negative_dist_calib, q_div, div=True)
        print(f'q_div: {q_div}, avg_len_single_new_div_val: {length_div_calib}, avg_cov_after_single_new_div_val: {coverage_div_calib}')
            
        # cal avg new len and cov test set
        length_div_test, coverage_div_test = calc_stats_new_method(target_test, mu_test, positive_dist_test, negative_dist_test, q_div, div=True)
        print(f'q_div: {q_div}, avg_len_single_new_div_test: {length_div_test}, avg_cov_after_single_new_div_test: {coverage_div_test}')
            
        q_all_new_div.append(get_float(q_div))
        avg_len_new_div.append(get_float(length_div_test))
        avg_cov_new_div.append(get_float(coverage_div_test))
           
        # CP 
            
        q = calc_optimal_q(target_calib, mu_calib, sd_calib, alpha)
                     
        
        valid_length, valid_coverage = calc_stats(q, target_calib, mu_calib, sd_calib)   
        test_length, test_coverage = calc_stats(q, target_test, mu_test, sd_test)
        
        
        # GC
            
        q_gc = calc_optimal_q(target_calib, mu_calib, sd_calib, alpha, gc=True)
                     
        valid_length_gc, valid_coverage_gc = calc_stats(q_gc, target_calib, mu_calib, sd_calib)
        test_length_gc, test_coverage_gc = calc_stats(q_gc, target_test, mu_test, sd_test)

    
        q_all.append(q)
        avg_len_all.append(test_length)
        avg_cov_all.append(test_coverage)
        
        q_all_gc.append(q_gc)
        avg_len_all_gc.append(test_length_gc)
        avg_cov_all_gc.append(test_coverage_gc)
        
        
        # calc optimal_q for new method
        mu_right_calib = positive_dist_calib - right_d_calib
        mu_left_calib = negative_dist_calib + left_d_calib
        mu_right_test = positive_dist_test - right_d_test
        mu_left_test = negative_dist_test + left_d_test
        
        q_mu = calc_opt_q_new_method_my_mu(target_calib, mu_right_calib, mu_left_calib, right_d_calib, left_d_calib, alpha)
        mu_length_calib, mu_coverage_calib = calc_stats_new_method_my_mu(target_calib, mu_right_calib, mu_left_calib, right_d_calib, left_d_calib, q_mu)
        mu_length_test, mu_coverage_test = calc_stats_new_method_my_mu(target_test, mu_right_test, mu_left_test, right_d_test, left_d_test, q_mu)
        print(f'q_mu: {q_mu}, avg_len_single_new_mu_val: {mu_length_calib}, avg_cov_after_single_new_mu_val: {mu_coverage_calib}')
        print(f'q_mu: {q_mu}, avg_len_single_new_mu_test: {mu_length_test}, avg_cov_after_single_new_mu_test: {mu_coverage_test}')
        q_mu_all.append(get_float(q_mu))
        avg_len_mu_all.append(get_float(mu_length_test))
        avg_cov_mu_all.append(get_float(mu_coverage_test))
        
        
    print(q_all)
    print(avg_len_all)
    print(avg_cov_all)
    print(q_all_gc)
    print(avg_len_all_gc)
    print(avg_cov_all_gc)

    # Define the output file path
    resutls_dir_path = get_dir_results(one_output=one_output)
    output_file = f"{dataset}_dataset_model_{base_model}_alpha_{alpha}_level_{level}_iterations_{iters}_lambda_{lambda_param}{'_x' if pred_x else ''}{'_y' if pred_y else ''}{'_normalize' if normalize else ''}_after.txt"
    
    # Open the file in append mode
    with open(f'{resutls_dir_path}/{output_file}', "w") as f:
        # Print and save CP metrics
        print(f'q CP mean: {statistics.mean(q_all)}, q CP std: {statistics.stdev(q_all)}')
        f.write(f'q CP mean: {statistics.mean(q_all)}, q CP std: {statistics.stdev(q_all)}\n')
        
        print(f'avg_len CP mean: {statistics.mean(avg_len_all)}, avg_len CP std: {statistics.stdev(avg_len_all)}')
        f.write(f'avg_len CP mean: {statistics.mean(avg_len_all)}, avg_len CP std: {statistics.stdev(avg_len_all)}\n')
        
        print(f'avg_cov CP mean: {statistics.mean(avg_cov_all)}, avg_cov CP std: {statistics.stdev(avg_cov_all)}')
        f.write(f'avg_cov CP mean: {statistics.mean(avg_cov_all)}, avg_cov CP std: {statistics.stdev(avg_cov_all)}\n')
        
        # Print and save GC metrics
        print(f'q GC mean: {statistics.mean(q_all_gc)}, q GC std: {statistics.stdev(q_all_gc)}')
        f.write(f'q GC mean: {statistics.mean(q_all_gc)}, q GC std: {statistics.stdev(q_all_gc)}\n')
        
        print(f'avg_len GC mean: {statistics.mean(avg_len_all_gc)}, avg_len GC std: {statistics.stdev(avg_len_all_gc)}')
        f.write(f'avg_len GC mean: {statistics.mean(avg_len_all_gc)}, avg_len GC std: {statistics.stdev(avg_len_all_gc)}\n')
        
        print(f'avg_cov GC mean: {statistics.mean(avg_cov_all_gc)}, avg_cov GC std: {statistics.stdev(avg_cov_all_gc)}')
        f.write(f'avg_cov GC mean: {statistics.mean(avg_cov_all_gc)}, avg_cov GC std: {statistics.stdev(avg_cov_all_gc)}\n')
        
        print(f'q new addtive mean: {statistics.mean(q_all_new_addtive)}, q new addtive std: {statistics.stdev(q_all_new_addtive)}')
        f.write(f'q new addtive mean: {statistics.mean(q_all_new_addtive)}, q new addtive std: {statistics.stdev(q_all_new_addtive)}\n')
        
        print(f'avg_len new addtive mean: {statistics.mean(avg_len_new_addtive)}, avg_len new addtive std: {statistics.stdev(avg_len_new_addtive)}')
        f.write(f'avg_len new addtive mean: {statistics.mean(avg_len_new_addtive)}, avg_len new addtive std: {statistics.stdev(avg_len_new_addtive)}\n')

        print(f'avg_cov new addtive mean: {statistics.mean(avg_cov_new_addtive)}, avg_cov new addtive std: {statistics.stdev(avg_cov_new_addtive)}')
        f.write(f'avg_cov new addtive mean: {statistics.mean(avg_cov_new_addtive)}, avg_cov new addtive std: {statistics.stdev(avg_cov_new_addtive)}\n')
        
        print(f'q new div mean: {statistics.mean(q_all_new_div)}, q new div std: {statistics.stdev(q_all_new_div)}')
        f.write(f'q new div mean: {statistics.mean(q_all_new_div)}, q new div std: {statistics.stdev(q_all_new_div)}\n')
        
        print(f'avg_len new div mean: {statistics.mean(avg_len_new_div)}, avg_len new div std: {statistics.stdev(avg_len_new_div)}')
        f.write(f'avg_len new div mean: {statistics.mean(avg_len_new_div)}, avg_len new div std: {statistics.stdev(avg_len_new_div)}\n')
        
        print(f'avg_cov new div mean: {statistics.mean(avg_cov_new_div)}, avg_cov new div std: {statistics.stdev(avg_cov_new_div)}')
        f.write(f'avg_cov new div mean: {statistics.mean(avg_cov_new_div)}, avg_cov new div std: {statistics.stdev(avg_cov_new_div)}\n')
               
               
        print(f'q mu mean: {statistics.mean(q_mu_all)}, q mu std: {statistics.stdev(q_mu_all)}')
        f.write(f'q mu mean: {statistics.mean(q_mu_all)}, q mu std: {statistics.stdev(q_mu_all)}\n')
        
        print(f'avg_len mu mean: {statistics.mean(avg_len_mu_all)}, avg_len mu std: {statistics.stdev(avg_len_mu_all)}')
        f.write(f'avg_len mu mean: {statistics.mean(avg_len_mu_all)}, avg_len mu std: {statistics.stdev(avg_len_mu_all)}\n')
        
        print(f'avg_cov mu mean: {statistics.mean(avg_cov_mu_all)}, avg_cov mu std: {statistics.stdev(avg_cov_mu_all)}')
        f.write(f'avg_cov mu mean: {statistics.mean(avg_cov_mu_all)}, avg_cov mu std: {statistics.stdev(avg_cov_mu_all)}\n')
           
        # Print and save additional info
        print(f"{dataset}, {base_model}, {alpha}, {level}")
        f.write(f"{dataset}, {base_model}, {alpha}, {level}\n")
    
  
def get_float(x):
    try:
        return x.item()
    except:
        return x
  
def to_pil_and_resize(x, scale):
    w, h, _ = x.shape
    new_size = (int(w * scale), int(h * scale))

    trans_always1 = [
        transforms.ToPILImage(),
        transforms.Resize(new_size),
    ]

    trans = transforms.Compose(trans_always1)
    x = trans(x)
    return x
    


    
    
if __name__ == '__main__':
    main()