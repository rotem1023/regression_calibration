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
from models import BreastPathQModel, DistancePredictor, DistancePredictorOneOutput, DistNewModel, BreastPathQModel3Heads
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





def calc_stats(q, target, mu, left_sd, right_sd):
    lower = mu - q * left_sd
    upper = mu + q * right_sd
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
    
    

def calc_optimal_q(target_calib, mu_calib, left_sd_calib, right_sd_calib, alpha):
    bigger_index = target_calib > mu_calib
    target_calib_bigger = target_calib[bigger_index]
    mu_calib_bigger = mu_calib[bigger_index]
    right_sd_calib_bigger = right_sd_calib[bigger_index]
    q_bigger = torch.quantile((target_calib_bigger - mu_calib_bigger) / right_sd_calib_bigger, 1 - alpha)
    
    
    target_calib_smaller = target_calib[~bigger_index]
    mu_calib_smaller = mu_calib[~bigger_index]
    left_sd_calib_smaller = left_sd_calib[~bigger_index]
    q_smaller = torch.quantile((mu_calib_smaller - target_calib_smaller) / left_sd_calib_smaller, 1 - alpha)
    
    s_t = torch.where(target_calib < mu_calib, (mu_calib - target_calib) / left_sd_calib, (target_calib - mu_calib) / right_sd_calib)
    s_t_sorted, _ = torch.sort(s_t, dim=0)
    q_index = math.ceil((len(s_t_sorted)) * (1 - alpha))
    q = s_t_sorted[q_index].item()   
    return q


def get_saved_dir(results_dir, dataset, base_model, dist_model, loss,group,  level = None):
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
    return cur_dir

    

def save_arrays(results_dir, dataset, base_model, dist_model, loss, group, level, y, mu, sd_left, sd_right):
    saved_dir = get_saved_dir(results_dir=results_dir, dataset=dataset, base_model=base_model, dist_model=dist_model, loss = loss, group = group, level=level)
    np.save(f'{saved_dir}/y.npy', y.cpu().numpy())
    np.save(f'{saved_dir}/mu.npy', mu.cpu().numpy())
    np.save(f'{saved_dir}/sd_left.npy', sd_left.cpu().numpy())  
    np.save(f'{saved_dir}/sd_right.npy', sd_right) 

def load_arrays(results_dir, dataset, base_model, dist_model, loss, group, level):
    saved_dir = get_saved_dir(results_dir=results_dir, dataset=dataset, base_model=base_model, dist_model=dist_model, loss = loss, group = group, level=level)
    y = np.load(f'{saved_dir}/y.npy')
    mu = np.load(f'{saved_dir}/mu.npy')
    sd_left = np.load(f'{saved_dir}/sd_left.npy')  
    sd_rgiht = np.load(f'{saved_dir}/sd_right.npy') 
    return torch.from_numpy(y), torch.from_numpy(mu), torch.from_numpy(sd_left), torch.from_numpy(sd_rgiht)
    
def modify_predicted_distances(predicted_distances, probs):
    first_dim_results = predicted_distances * probs
    second_dim_results = predicted_distances * (1 - probs)
    return torch.stack([first_dim_results, second_dim_results], dim=1).squeeze(-1)

def get_arrays(data_loader, dist_model, device):
    targets_s = []
    mu_s = []
    left_log_var_s = []
    right_log_var_s = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)

            mu, left_log_var, right_log_var = dist_model(data, dropout=False, mc_dropout=False, test=False)
            targets_s.append(target.detach())
            mu_s.append(mu.detach())
            left_log_var_s.append(left_log_var.detach())
            right_log_var_s.append(right_log_var.detach())
    return torch.cat(mu_s).cpu(), torch.cat(left_log_var_s).cpu(), torch.cat(right_log_var_s).cpu(), torch.cat(targets_s).cpu() 


            
                    
    


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
    resutls_dir_path = '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results_new/asym'
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
    base_model_dist = 'resnet50'
    assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
    device = torch.device("cuda:2")
    dataset = 'lumbar'
    loss = 'gaussian'
    pred_x = False
    pred_y = False
    one_output = False
    load_results = False
    iters = 20
    level = 5
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
            model = BreastPathQModel3Heads(base_model, in_channels=3, out_channels=1,
                             pretrained=True).to(device) 
            checkpoint = torch.load(f"/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots_asym/efficientnetb4_lumbar_L{level}_snapshot_best.pth.tar", map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"epoch: {checkpoint['epoch']}")
            model.eval()  
                 
            data_set_valid_original = LumbarDataset(level=level, mode='val', augment=False, pred_x=pred_x, pred_y=pred_x, scale=0.5)
            data_set_test_original = LumbarDataset(level=level, mode='test', augment=False, pred_x=pred_x, pred_y=pred_x, scale=0.5)
        elif dataset == 'boneage':
            resize_to = (256, 256)
            data_set_valid_original = BoneAgeDataset(group='valid', augment=False, resize_to=resize_to)
            data_set_test_original = BoneAgeDataset(group='test', augment=False, resize_to=resize_to)
            model = load_trained_models.get_model_boneage(base_model, None, device, loss=loss)
        else:
            assert False
    
    
        assert len(data_set_valid_original) > 0
        assert len(data_set_test_original) > 0
        print(len(data_set_valid_original))
        print(len(data_set_test_original))
            
        calib_loader = torch.utils.data.DataLoader(data_set_valid_original, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(data_set_test_original, batch_size=batch_size, shuffle=False)
        y_p_calib_original, log_var_left_calib_original , log_var_right_calib_original, targets_calib_original = get_arrays(calib_loader, model, device)
        y_p_test_original, log_var_left_test_original , log_var_right_test_original, targets_test_original  = get_arrays(test_loader, model,  device)
        sd_left_calib_original = log_var_left_calib_original.exp().sqrt()
        sd_right_calib_original = log_var_right_calib_original.exp().sqrt()
        sd_left_test_original = log_var_left_test_original.exp().sqrt()
        sd_right_test_original = log_var_right_test_original.exp().sqrt()

        save_arrays(results_dir = results_dir, dataset = dataset, base_model = base_model, dist_model = base_model_dist, loss = loss, group = 'valid', level = cur_level, y = targets_calib_original, mu=y_p_calib_original, sd_left=sd_left_calib_original, sd_right=sd_right_calib_original)
        save_arrays(results_dir = results_dir, dataset = dataset, base_model = base_model, dist_model = base_model_dist, loss = loss, group = 'test', level = cur_level, y = targets_test_original, mu=y_p_test_original, sd_left=sd_left_test_original, sd_right=sd_right_test_original)
    else:
        targets_calib_original, y_p_calib_original, sd_left_calib_original, sd_right_calib_original = load_arrays(results_dir = results_dir, dataset = dataset, base_model = base_model, dist_model = base_model_dist, loss = loss, group = 'valid', level = cur_level)
        targets_test_original, y_p_test_original, sd_left_test_original, sd_right_test_original = load_arrays(results_dir = results_dir, dataset = dataset, base_model = base_model, dist_model = base_model_dist, loss = loss, group = 'test', level = cur_level)
        
    

    

    # Calibration and test arrays (from your original code)
    calib_arrays = [
        y_p_calib_original,  
        targets_calib_original, 
        sd_left_calib_original,
        sd_right_calib_original,
    ]

    test_arrays = [
        y_p_test_original, 
        targets_test_original, 
        sd_left_test_original,
        sd_right_test_original,
    ]
    

    

    q_all = []
    avg_len_all = []
    avg_cov_all = []


    for j in range(iters):
        print(f'Iter: {j}')
        y_p_calib = []
        targets_calib = []
        
        if mix_indices:
            calib_shuffled, test_shuffled = shuffle_arrays(calib_arrays, test_arrays)
            y_p_calib, targets_calib, left_sd_calib, right_sd_calib = calib_shuffled
            y_p_test, targets_test, left_sd_test, right_sd_test = test_shuffled
        
                    
        # validation set   
        y_p_calib = y_p_calib.clamp(0, 1).unsqueeze(1)
        mu_calib = y_p_calib.mean(dim=1)

        if dataset== 'boneage':
            target_calib = targets_calib
        else:
            target_calib = targets_calib.unsqueeze(1)
            
        
        # test set
                                 
        y_p_test = y_p_test.clamp(0, 1).unsqueeze(1)
        mu_test = y_p_test.mean(dim=1)
        if dataset== 'boneage':
            target_test = targets_test
        else:
            target_test = targets_test.unsqueeze(1)

      
           
        # CP 
            
        q = calc_optimal_q(target_calib, mu_calib, left_sd_calib, right_sd_calib, alpha)
                     
        
        valid_length, valid_coverage = calc_stats(q, target_calib, mu_calib, left_sd_calib, right_sd_calib)   
        test_length, test_coverage = calc_stats(q, target_test, mu_test, left_sd_test, right_sd_test)
        
        print(f'q: {q}')
        print(f'valid_length: {valid_length}, valid_coverage: {valid_coverage}')
        print(f'test_length: {test_length}, test_coverage: {test_coverage}')
        
    

    
        q_all.append(q)
        avg_len_all.append(test_length)
        avg_cov_all.append(test_coverage)

        


        
        
    print(q_all)
    print(avg_len_all)
    print(avg_cov_all)


    # Define the output file path
    resutls_dir_path = get_dir_results(one_output=one_output)
    output_file = f"{dataset}_dataset_model_{base_model}_alpha_{alpha}_level_{level}_iterations_{iters}_{'_x' if pred_x else ''}{'_y' if pred_y else ''}.txt"
    
    # Open the file in append mode
    with open(f'{resutls_dir_path}/{output_file}', "w") as f:
        # Print and save CP metrics
        print(f'q CP asym  mean: {statistics.mean(q_all)}, q CP std: {statistics.stdev(q_all)}')
        f.write(f'q CP asym mean: {statistics.mean(q_all)}, q CP std: {statistics.stdev(q_all)}\n')
        
        print(f'avg_len CP asym mean: {statistics.mean(avg_len_all)}, avg_len CP std: {statistics.stdev(avg_len_all)}')
        f.write(f'avg_len CP asym mean: {statistics.mean(avg_len_all)}, avg_len CP std: {statistics.stdev(avg_len_all)}\n')
        
        print(f'avg_cov CP asym mean: {statistics.mean(avg_cov_all)}, avg_cov CP std: {statistics.stdev(avg_cov_all)}')
        f.write(f'avg_cov CP asym mean: {statistics.mean(avg_cov_all)}, avg_cov CP std: {statistics.stdev(avg_cov_all)}\n')
        
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