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
from models import BreastPathQModel, DistancePredictor
from glob import glob
import statistics
import math


    
    

# CP

def calc_optimal_q(target_calib, mu_calib, sd_calib, alpha, gc=False):

    s_t = torch.abs(target_calib-mu_calib) / sd_calib
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
    lower = mu - q * sd
    upper = mu + q * sd
    length = torch.mean(abs(upper - lower))
    coverage = avg_cov(lower, upper, target)
    return length, coverage

def avg_cov(lower, upper, target):
    in_the_range = torch.sum((target  >= lower) & (target  <= upper)).item()
    coverage = in_the_range / len(target) * 100
    return coverage



def get_arrays(data_loader, model, device):
    y_p_s = []
    vars_s = []
    logvars_s = []
    targets_s = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)

            y_p, logvar, var_bayesian = model(data, dropout=False, mc_dropout=False, test=False)

            y_p_s.append(y_p.detach())
            vars_s.append(var_bayesian.detach())
            logvars_s.append(logvar.detach())
            targets_s.append(target.detach())

                            
                    
    return torch.cat(y_p_s).cpu(), torch.cat(vars_s).cpu(), torch.cat(logvars_s).cpu(), torch.cat(targets_s).cpu()     
    
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
    models_dir = '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots'
    assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
    device = torch.device("cuda:0")
    iters = 20
    level = 1
    alpha = 0.1
    
    print(f'alpha: {alpha}, level: {level}, base_model: {base_model}, mix_indices: {mix_indices}, save_params: {save_params}, load_params: {load_params}, calc_mean: {calc_mean}, save_test: {save_test}, load_test: {load_test}')
    
    model = BreastPathQModel(base_model, out_channels=1).to(device)

    # TODO: load checkpoint 
    # checkpoint_path = glob(f"/home/dsi/frenkel2/regression_calibration/models/{base_model}_gaussian_endovis_199_new.pth.tar")[0]
    # checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\{base_model}_gaussian_endovis_199_new.pth.tar")[0]
    
    checkpoint = torch.load(f'{models_dir}/{base_model}_gaussian_lumbar_L{level}_best.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"epoch: {checkpoint['epoch']}")
    
    batch_size = 64


    data_set_valid_original = LumbarDataset(level=level, mode='val', augment=False, scale=0.5)
    data_set_test_original = LumbarDataset(level=level, mode='test', augment=False, scale=0.5)
    
    assert len(data_set_valid_original) > 0
    assert len(data_set_test_original) > 0
    print(len(data_set_valid_original))
    print(len(data_set_test_original))
        
    calib_loader = torch.utils.data.DataLoader(data_set_valid_original, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(data_set_test_original, batch_size=batch_size, shuffle=False)
    
    y_p_calib_original, vars_calib_original, logvars_calib_original, targets_calib_original = get_arrays(calib_loader, model, device)
    y_p_test_original, vars_test_original, logvars_test_original, targets_test_original = get_arrays(test_loader, model, device)
    
    # save test arrays
    results_dir = "/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results/predictions/"
    
    np.save(f'{results_dir}/lumbar_dataset_model_{base_model}_level{level}_y_p_test_original.npy', y_p_test_original.cpu().numpy())
    np.save(f'{results_dir}/lumbar_dataset_model_{base_model}_level{level}_logvars_test_original.npy', logvars_test_original.cpu().numpy())
    np.save(f'{results_dir}/lumbar_dataset_model_{base_model}_level{level}_targets_test_original.npy', targets_calib_original.cpu().numpy())  
    np.save(f'{results_dir}/lumbar_dataset_model_{base_model}_level{level}_y_p_calib_original.npy', y_p_calib_original.cpu().numpy()) 
    np.save(f'{results_dir}/lumbar_dataset_model_{base_model}_level{level}_logvars_calib_original.npy', logvars_calib_original.cpu().numpy())
    np.save(f'{results_dir}/lumbar_dataset_model_{base_model}_level{level}_targets_calib_original.npy', targets_test_original.cpu().numpy())

    
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
    
    avg_len_valid_all = []
    avg_cov_valid_all = []
    avg_len_valid_all_gc = []
    avg_cov_valid_all_gc = []
    
    


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
        
                    
        # validation set   
        y_p_calib = y_p_calib.clamp(0, 1).unsqueeze(1)
        mu_calib = y_p_calib.mean(dim=1)
        var_calib = vars_calib
        logvars_calib = logvars_calib
        logvar_calib = logvars_calib.mean(dim=1).unsqueeze(1)
        var_calib  = logvar_calib.exp()
        sd_calib = var_calib.sqrt()
        target_calib = targets_calib.unsqueeze(1)




        
        # test set
                                 
        y_p_test = y_p_test.clamp(0, 1).unsqueeze(1)
        mu_test = y_p_test.mean(dim=1)
        logvars_test = logvars_test
        logvar_test = logvars_test.mean(dim=1).unsqueeze(1)
        var_test = logvar_test.exp()
        sd_test = var_test.sqrt()
        target_test = targets_test.unsqueeze(1)
            
        q = calc_optimal_q(target_calib, mu_calib, sd_calib, alpha)
        
        valid_length, valid_coverage = calc_stats(q, target_calib, mu_calib, sd_calib)
        test_length, test_coverage = calc_stats(q, target_test, mu_test, sd_test)
                     
            
        q_gc = calc_optimal_q(target_calib, mu_calib, sd_calib, alpha, gc=True)
                     
        valid_length_gc, valid_coverage_gc = calc_stats(q_gc, target_calib, mu_calib, sd_calib)
        test_length_gc, test_coverage_gc = calc_stats(q_gc, target_test, mu_test, sd_test)
            
        print(f'q: {q}, q_gc: {q_gc}')
        print(f'valid_length: {valid_length}, valid_coverage: {valid_coverage}')
        print(f'test_length: {test_length}, test_coverage: {test_coverage}')
        print(f'valid_length_gc: {valid_length_gc}, valid_coverage_gc: {valid_coverage_gc}')
        print(f'test_length_gc: {test_length_gc}, test_coverage_gc: {test_coverage_gc}')
            

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
        
    print(f"q cp: {q_all}")
    print(f"q gc: {q_all_gc}")
    print(f"avg_len cp: {avg_len_all}")
    print(f"avg_len gc: {avg_len_all_gc}")
    print(f"avg_cov cp: {avg_cov_all}")
    print(f"avg_cov gc: {avg_cov_all_gc}")


    # Define the output file path
    output_dir= '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results'
    output_file = f"lumbar_dataset_model_{base_model}_alpha_{alpha}_level_{level}_iterations_{iters}_after.txt"

    # Open the file in append mode
    with open(f'{output_dir}/{output_file}', "w") as f:
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
        
        # save validation results
        f.write(f"Validation results:\n")
        print(f"avg_len validation mean: {statistics.mean(avg_len_valid_all)}, avg_len validation std: {statistics.stdev(avg_len_valid_all)}")
        f.write(f"avg_len validation mean: {statistics.mean(avg_len_valid_all)}, avg_len validation std: {statistics.stdev(avg_len_valid_all)}\n")
        
        print(f"avg_cov validation mean: {statistics.mean(avg_cov_valid_all)}, avg_cov validation std: {statistics.stdev(avg_cov_valid_all)}")
        f.write(f"avg_cov validation mean: {statistics.mean(avg_cov_valid_all)}, avg_cov validation std: {statistics.stdev(avg_cov_valid_all)}\n")
        
        print(f"avg_len validation mean GC: {statistics.mean(avg_len_valid_all_gc)}, avg_len validation std GC: {statistics.stdev(avg_len_valid_all_gc)}")
        f.write(f"avg_len validation mean GC: {statistics.mean(avg_len_valid_all_gc)}, avg_len validation std GC: {statistics.stdev(avg_len_valid_all_gc)}\n")
        
        print(f"avg_cov validation mean GC: {statistics.mean(avg_cov_valid_all_gc)}, avg_cov validation std GC: {statistics.stdev(avg_cov_valid_all_gc)}")
        f.write(f"avg_cov validation mean GC: {statistics.mean(avg_cov_valid_all_gc)}, avg_cov validation std GC: {statistics.stdev(avg_cov_valid_all_gc)}\n")        
        
                 
        # Print and save additional info
        print(f"lumbar, {base_model}, {alpha}, {level}")
        f.write(f"lumbar, {base_model}, {alpha}, {level}\n")
    
  
def get_float(x):
    try:
        return x.item()
    except:
        return x
  
    


    
    
if __name__ == '__main__':
    main()