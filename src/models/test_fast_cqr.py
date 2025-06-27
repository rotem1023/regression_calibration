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
from cqr_model import BreastPathQModel
from glob import glob
import statistics
import math
import load_trained_models
from data_generator_brain import BrainDatasetTest, BrainDatasetVal 
from functools import reduce
from operator import mul

    
    

    
# CQR

def calc_optimal_q(target_calib, mu_calib, alpha=0.1):
    y_lower = mu_calib[:,0]
    y_upper = mu_calib[:,-1]
    error_low = y_lower - target_calib
    error_high = target_calib - y_upper
    err = torch.maximum(error_high, error_low)
    err, _ = torch.sort(err, 0)
    index = int(math.ceil((1 - alpha) * (err.shape[0] + 1))) - 1
    index = min(max(index, 0), err.shape[0] - 1)
    q = err[index]
    
    return q.item()



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
    coverage = in_the_range / len(y_test) * 100
    return ((y_test >= y_lower) & (y_test <= y_upper)), torch.mean(abs(y_upper - y_lower))

def compute_coverage_len(targets, preds, q_s):
    coverages, lengths = [], []
    for i in range(targets.ndim):
        cur_tgrets = targets[:, i]
        cur_preds = preds[i]
        cur_q = q_s[i]
        y_lower = cur_preds[:, 0] - cur_q
        y_upper = cur_preds[:, 1] + cur_q
        cur_coverage, cur_length = compute_in_range_len_one_dim(cur_tgrets, y_lower, y_upper)
        coverages.append(cur_coverage)
        lengths.append(cur_length)

    overall_cvg = torch.stack(coverages).all(dim=0)

    # Convert boolean result to int (0/1)
    overall_cvg = overall_cvg.to(dtype=torch.uint8)   
    
    coverage = torch.sum(overall_cvg) / len(overall_cvg)
    length = reduce(mul, lengths)
    return  torch.mean(length).item(), coverage * 100

def get_scaler_conformal(target, preds, alpha):
    """
    Tune single scaler for the model (using the validation set) with cross-validation on NLL
    """
    dims = preds.shape[0]
    actual_alpha = alpha / dims    
    printed_type = 'CQR'
    q_s = []
    for i in range(dims):
        dim_preds = preds[i]
        dim_target = target[:, i]
        print(f'Calculating {printed_type} for dim {i}')
        # Calculate optimal q for this dimension
        q = calc_optimal_q(dim_target, dim_preds, alpha=actual_alpha)
        q_s.append(q)
            
    # Calculate optimal q
    return q_s

def avg_cov(mu, q, target, before=False):
    if before:
        in_the_range = torch.sum((target.squeeze(-1)  >= mu[:, 0]) & (target.squeeze(-1)  <= mu[:, 1]))
    else:
        in_the_range = torch.sum((target.squeeze(-1)  >= (mu[:, 0] - q)) & (target.squeeze(-1)  <= (mu[:, 1] + q)))
    coverage = in_the_range / len(target) * 100
    return coverage


def create_final_preds(t_p_s):
    preds = []
    for i in range(len(t_p_s[0])):
        dim_preds = []
        for j in range(len(t_p_s)):
            dim_preds.append(t_p_s[j][i].detach().cpu())
        dim_preds = torch.cat(dim_preds, dim=1).clamp(0, 1).permute(1,0,2).mean(dim=1)
        preds.append(dim_preds)
    return torch.stack(preds)
    


def get_arrays(data_loader, model, device):
    t_p_s = []
    targets_s = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)

            t_p = model(data, dropout=True, mc_dropout=True, test=True)

            t_p_s.append(t_p)

            targets_s.append(target.detach()) 


        targets = torch.cat(targets_s).cpu()
        final_preds = create_final_preds(t_p_s)
                            
                    
    return final_preds, targets    
    
import numpy as np
import torch

def shuffle_arrays(calib_arrays, test_arrays):
    calib_preds, calib_targets = calib_arrays
    test_preds, test_targets = test_arrays

    # Check that shapes match in dimensions
    assert calib_preds.shape[0] == test_preds.shape[0], "Mismatch in num_dims of prediction arrays"
    assert calib_preds.shape[2] == 2, "Predictions must have lower and upper bounds"
    assert calib_targets.shape[1] == test_targets.shape[1], "Mismatch in num_dims of target arrays"

    # Concatenate along the num_predictions axis
    all_preds = np.concatenate([calib_preds, test_preds], axis=1)  # shape: [num_dims, total_preds, 2]
    all_targets = np.concatenate([calib_targets, test_targets], axis=0)  # shape: [total_preds, num_dims]

    # Shuffle indices
    total_preds = all_preds.shape[1]
    indices = np.arange(total_preds)
    np.random.shuffle(indices)

    # Apply shuffle
    shuffled_preds = all_preds[:, indices, :]
    shuffled_targets = all_targets[indices]

    # Split back
    calib_size = calib_preds.shape[1]
    new_calib_preds = shuffled_preds[:, :calib_size, :]
    new_test_preds = shuffled_preds[:, calib_size:, :]

    new_calib_targets = shuffled_targets[:calib_size]
    new_test_targets = shuffled_targets[calib_size:]

    return (torch.from_numpy(new_calib_preds), torch.from_numpy(new_calib_targets)), (torch.from_numpy(new_test_preds), torch.from_numpy(new_test_targets))   

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
    models_dir = '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots/cqr'
    assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
    device = torch.device("cuda:0")
    dataset = 'lumbar'
    iters = 20
    level = 1
    alpha = 0.1
    load_preds = False
    
    print(f'Running CQR for model {base_model} with alpha {alpha} and level {level}, {iters} iterations')
    
    
    results_dir = "/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results/predictions/cqr/dims"
    if not load_preds:
        model = BreastPathQModel(base_model, out_channels=2).to(device)

        # checkpoint_path = glob(f"/home/dsi/frenkel2/regression_calibration/models/{base_model}_gaussian_endovis_199_new.pth.tar")[0]
        # checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\{base_model}_gaussian_endovis_199_new.pth.tar")[0]
        if dataset == 'brain':
                checkpoint = torch.load(f'{models_dir}/{base_model}_{dataset}_L1_alpha_{alpha}_cqr_best.pth.tar', map_location=device)
        else:
            checkpoint = torch.load(f'{models_dir}/{base_model}_lumbar_L{level}_alpha_{alpha}_cqr_dims.pth.tar', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"epoch: {checkpoint['epoch']}")
        
        batch_size = 64

        if dataset =='brain':
            data_set_valid_original = BrainDatasetVal(model=base_model)
            data_set_test_original = BrainDatasetTest(model=base_model)
        else:
            data_set_valid_original = LumbarDataset(level=level, mode='val', augment=False, scale=0.5)
            data_set_test_original = LumbarDataset(level=level, mode='test', augment=False, scale=0.5)
        
        assert len(data_set_valid_original) > 0
        assert len(data_set_test_original) > 0
        print(len(data_set_valid_original))
        print(len(data_set_test_original))
            
        calib_loader = torch.utils.data.DataLoader(data_set_valid_original, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(data_set_test_original, batch_size=batch_size, shuffle=False)
        
        y_p_calib_original, targets_calib_original = get_arrays(calib_loader, model, device)
        y_p_test_original, targets_test_original = get_arrays(test_loader, model, device)
        
        # save arrays
        os.makedirs(results_dir, exist_ok=True)
        np.save(f'{results_dir}/{dataset}_dataset_cqr_model_{base_model}_alpha_{alpha}_level_{level}_y_p_calib_original.npy', y_p_calib_original.cpu().numpy())
        np.save(f'{results_dir}/{dataset}_dataset_cqr_model_{base_model}_alpha_{alpha}_level_{level}_targets_calib_original.npy', targets_calib_original.cpu().numpy())
        np.save(f'{results_dir}/{dataset}_dataset_cqr_model_{base_model}_alpha_{alpha}_level_{level}_y_p_test_original.npy', y_p_test_original.cpu().numpy())
        np.save(f'{results_dir}/{dataset}_dataset_cqr_model_{base_model}_alpha_{alpha}_level_{level}_targets_test_original.npy', targets_test_original.cpu().numpy())
    else:
        # Load arrays
        y_p_calib_original = torch.from_numpy(np.load(f'{results_dir}/{dataset}_dataset_cqr_model_{base_model}_alpha_{alpha}_level_{level}_y_p_calib_original.npy'))
        targets_calib_original = torch.from_numpy(np.load(f'{results_dir}/{dataset}_dataset_cqr_model_{base_model}_alpha_{alpha}_level_{level}_targets_calib_original.npy'))
        y_p_test_original = torch.from_numpy(np.load(f'{results_dir}/{dataset}_dataset_cqr_model_{base_model}_alpha_{alpha}_level_{level}_y_p_test_original.npy'))
        targets_test_original = torch.from_numpy(np.load(f'{results_dir}/{dataset}_dataset_cqr_model_{base_model}_alpha_{alpha}_level_{level}_targets_test_original.npy'))    
    
    
    # Calibration and test arrays (from your original code)
    calib_arrays = [
        y_p_calib_original.cpu(),
        targets_calib_original.cpu()
    ]

    test_arrays = [
        y_p_test_original.cpu(), 
        targets_test_original.cpu()
    ]

    q_all = []
    len_valid_sets = []
    cov_valid_sets = []
    q_all_gc = []
    len_test_sets = []
    cov_test_sets = []
    


    for j in range(iters):
        print(f'Iter: {j}')
        targets_calib = []
        
        if mix_indices:
            calib_shuffled, test_shuffled = shuffle_arrays(calib_arrays, test_arrays)
            t_p_calib, targets_calib = calib_shuffled
            t_p_test, targets_test = test_shuffled
            
            
        
        # validation set   
        t_p_calib = t_p_calib.clamp(0, 1)
        target_calib = targets_calib

            

            
        
        t_p_test_list = []
        target_test_list = []

        # test set
                                 
        t_p_test = t_p_test.clamp(0, 1)
        target_test = targets_test



        t_p_test_list.append(t_p_test)
        target_test_list.append(target_test)
                

        q = get_scaler_conformal(target_calib, t_p_calib, alpha)
        print("validation set")
        length_val, coverage_val = compute_coverage_len(target_calib, t_p_calib, q)
        print("test set")
        length_test, coverage_test = compute_coverage_len(target_test, t_p_test, q)    
    
        
        q_all.append(get_float(q))
        len_valid_sets.append(get_float(length_val))
        cov_valid_sets.append(get_float(coverage_val))
        
        len_test_sets.append(get_float(length_test))
        cov_test_sets.append(get_float(coverage_test))
        
    print(f" q's: {q_all}")
    print(f"valid len's {len_valid_sets}")
    print(f"valid coverage's {cov_valid_sets}")
    print(f"test coverages's {len_test_sets}")
    print(f"test coverage's {cov_test_sets}")

    # Define the output file path
    output_dir= '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results/cqr/dims'
    output_file = f"{dataset}_dataset_model_{base_model}_alpha_{alpha}_level_{level}_iterations_{iters}.txt"

    # Open the file in append mode
    with open(f'{output_dir}/{output_file}', "w") as f:
        # Print and save CP metrics
        # print(f'q mean: {statistics.mean(q_all)}, q std: {statistics.stdev(q_all)}')
        # f.write(f'q mean: {statistics.mean(q_all)}, q std: {statistics.stdev(q_all)}\n')
        
        print(f'avg_len valid mean: {statistics.mean(len_valid_sets)}, avg_len valid std: {statistics.stdev(len_valid_sets)}')
        f.write(f'avg_len valid  mean: {statistics.mean(len_valid_sets)}, avg_len valid std: {statistics.stdev(len_valid_sets)}\n')
        
        print(f'avg_cov valid mean: {statistics.mean(cov_valid_sets)}, avg_cov std: {statistics.stdev(cov_valid_sets)}')
        f.write(f'avg_cov valid mean: {statistics.mean(cov_valid_sets)}, avg_cov std: {statistics.stdev(cov_valid_sets)}\n')
        
        
        print(f'avg_len test mean: {statistics.mean(len_test_sets)}, avg_len  test std: {statistics.stdev(len_test_sets)}')
        f.write(f'avg_len test mean: {statistics.mean(len_test_sets)}, avg_len test std: {statistics.stdev(len_test_sets)}\n')
        
        print(f'avg_cov test mean: {statistics.mean(cov_test_sets)}, avg_cov test std: {statistics.stdev(cov_test_sets)}')
        f.write(f'avg_cov test mean: {statistics.mean(cov_test_sets)}, avg_cov test std: {statistics.stdev(cov_test_sets)}\n')
        
                 
        # Print and save additional info
        print(f"{dataset} cqr, {base_model}, {alpha}, {level}")
        f.write(f"{dataset} cqr, {base_model}, {alpha}, {level}\n")
    
  
def get_float(x):
    try:
        return x.item()
    except:
        return x
  

    


    
    
if __name__ == '__main__':
    main()