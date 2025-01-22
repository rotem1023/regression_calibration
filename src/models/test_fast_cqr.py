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


    
    

    
# CQR

def calc_optimal_q(target_calib, mu_calib, alpha=0.1):
    y_lower = mu_calib[:,0].unsqueeze(1)
    y_upper = mu_calib[:,-1].unsqueeze(1)
    error_low = y_lower - target_calib
    error_high = target_calib - y_upper
    err = torch.maximum(error_high, error_low)
    err, _ = torch.sort(err, 0)
    index = int(math.ceil((1 - alpha) * (err.shape[0] + 1))) - 1
    index = min(max(index, 0), err.shape[0] - 1)
    q = err[index]
    
    return q


def calc_stats(q, target, mu):
    length = torch.mean(abs((mu[:, 1] + q) - (mu[:, 0] - q)))
    coverage = avg_cov(mu, q, target.unsqueeze(1).mean(dim=1))
    print(f'Length: {length}, Coverage: {coverage}')
    return length, coverage

def get_scaler_conformal(target_calib, mu_calib, alpha):
    """
    Tune single scaler for the model (using the validation set) with cross-validation on NLL
    """
        
    printed_type = 'CQR'
            
    # Calculate optimal q
    q = calc_optimal_q(target_calib, mu_calib, alpha=alpha)
    print(f'q: {q}')
    return q

def avg_cov(mu, q, target, before=False):
    if before:
        in_the_range = torch.sum((target.squeeze(-1)  >= mu[:, 0]) & (target.squeeze(-1)  <= mu[:, 1]))
    else:
        in_the_range = torch.sum((target.squeeze(-1)  >= (mu[:, 0] - q)) & (target.squeeze(-1)  <= (mu[:, 1] + q)))
    coverage = in_the_range / len(target) * 100
    return coverage



def get_arrays(data_loader, model, device):
    t_p_s = []
    targets_s = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)

            t_p = model(data, dropout=False, mc_dropout=False, test=False)

            t_p_s.append(t_p.detach())

            targets_s.append(target.detach()) 

                            
                    
    return torch.cat(t_p_s), torch.cat(targets_s)     
    
import numpy as np
import torch

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
    models_dir = '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots/cqr'
    assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
    device = torch.device("cuda:0")
    iters = 20
    level = 2
    alpha = 0.05
    
    print(f'Running CQR for model {base_model} with alpha {alpha} and level {level}, {iters} iterations')
    
    model = BreastPathQModel(base_model, out_channels=2).to(device)

    # checkpoint_path = glob(f"/home/dsi/frenkel2/regression_calibration/models/{base_model}_gaussian_endovis_199_new.pth.tar")[0]
    # checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\{base_model}_gaussian_endovis_199_new.pth.tar")[0]
    checkpoint = torch.load(f'{models_dir}/{base_model}_lumbar_L{level}_alpha_{alpha}_cqr_new.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    batch_size = 64


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
    results_dir = "/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results/predictions/cqr"
    np.save(f'{results_dir}/lumbar_dataset_cqr_model_{base_model}_alpha_{alpha}_level_{level}_y_p_calib_original.npy', y_p_calib_original.cpu().numpy())
    np.save(f'{results_dir}/lumbar_dataset_cqr_model_{base_model}_alpha_{alpha}_level_{level}_targets_calib_original.npy', targets_calib_original.cpu().numpy())
    np.save(f'{results_dir}/lumbar_dataset_cqr_model_{base_model}_alpha_{alpha}_level_{level}_y_p_test_original.npy', y_p_test_original.cpu().numpy())
    np.save(f'{results_dir}/lumbar_dataset_cqr_model_{base_model}_alpha_{alpha}_level_{level}_targets_test_original.npy', targets_test_original.cpu().numpy())
    
    
    
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
        target_calib = targets_calib.unsqueeze(1)

            

            
        
        t_p_test_list = []
        target_test_list = []

        # test set
                                 
        t_p_test = t_p_test.clamp(0, 1)
        target_test = targets_test.unsqueeze(1)



        t_p_test_list.append(t_p_test)
        target_test_list.append(target_test)
                

        q = get_scaler_conformal(target_calib, t_p_calib, alpha)
        print("validation set")
        length_val, coverage_val = calc_stats(q, target_calib, t_p_calib)
        print("test set")
        length_test, coverage_test = calc_stats(q, target_test, t_p_test)    
    
        
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
    output_dir= '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results/cqr'
    output_file = f"lumbar_dataset_model_{base_model}_alpha_{alpha}_level_{level}_iterations_{iters}.txt"

    # Open the file in append mode
    with open(f'{output_dir}/{output_file}', "a") as f:
        # Print and save CP metrics
        print(f'q mean: {statistics.mean(q_all)}, q std: {statistics.stdev(q_all)}')
        f.write(f'q mean: {statistics.mean(q_all)}, q std: {statistics.stdev(q_all)}\n')
        
        print(f'avg_len valid mean: {statistics.mean(len_valid_sets)}, avg_len valid std: {statistics.stdev(len_valid_sets)}')
        f.write(f'avg_len valid  mean: {statistics.mean(len_valid_sets)}, avg_len valid std: {statistics.stdev(len_valid_sets)}\n')
        
        print(f'avg_cov valid mean: {statistics.mean(cov_valid_sets)}, avg_cov std: {statistics.stdev(cov_valid_sets)}')
        f.write(f'avg_cov valid mean: {statistics.mean(cov_valid_sets)}, avg_cov std: {statistics.stdev(cov_valid_sets)}\n')
        
        
        print(f'avg_len test mean: {statistics.mean(len_test_sets)}, avg_len  test std: {statistics.stdev(len_test_sets)}')
        f.write(f'avg_len test mean: {statistics.mean(len_test_sets)}, avg_len test std: {statistics.stdev(len_test_sets)}\n')
        
        print(f'avg_cov test mean: {statistics.mean(cov_test_sets)}, avg_cov test std: {statistics.stdev(cov_test_sets)}')
        f.write(f'avg_cov test mean: {statistics.mean(cov_test_sets)}, avg_cov test std: {statistics.stdev(cov_test_sets)}\n')
        
                 
        # Print and save additional info
        print(f"lumbar cqr, {base_model}, {alpha}, {level}")
        f.write(f"lumbar cqr, {base_model}, {alpha}, {level}\n")
    
  
def get_float(x):
    try:
        return x.item()
    except:
        return x
  

    


    
    
if __name__ == '__main__':
    main()