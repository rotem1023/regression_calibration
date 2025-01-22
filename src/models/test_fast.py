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
import load_trained_models


    
    

# CP

def calc_optimal_q(target_calib, mu_calib, uncert_calib, err_calib=None, alpha=0.1, gc=False, single=False):

    if single:
        s_t = torch.abs(target_calib-mu_calib)[:, 0].unsqueeze(-1) / uncert_calib
    else:
        s_t = torch.abs(target_calib-mu_calib) / uncert_calib
    if gc:
        # q = 1.64485 * torch.sqrt((s_t**2).mean()).item()
        # q = 1.64485 * s_t.median().item()
        S = (err_calib**2 / uncert_calib**2).mean().sqrt()
        # print(S)
        # q = 1.64485 * torch.sqrt((s_t**2).mean()).item()
        if alpha == 0.1:
            q = 1.64485 * S.item()
        elif alpha == 0.05:
            q = 1.95996 * S.item()
        else:
            print("Choose another value of alpha!! (0.1 / 0.05)")
    else:
        s_t_sorted, _ = torch.sort(s_t, dim=0)
        # q_index = math.ceil((len(s_t_sorted) + 1) * (1 - alpha))
        q_index = math.ceil((len(s_t_sorted)) * (1 - alpha))
        q = s_t_sorted[q_index].item()
        # q = torch.quantile(s_t, (1 - alpha))
    
    return q

# CP/GC prediction

def set_scaler_conformal(target_calib, mu_calib, uncert_calib, err_calib=None, log=True, gc=False, alpha=0.1):
    """
    Tune single scaler for the model (using the validation set) with cross-validation on NLL
    """
        
    if gc:
        printed_type = 'GC'
    else:
        printed_type = 'CP'
            
    # Calculate optimal q using GC
    q = calc_optimal_q(target_calib, mu_calib, uncert_calib, err_calib=err_calib, alpha=alpha, gc=gc)
    
    after_single_scaling_avg_len = avg_len(uncert_calib, q)
    print('Optimal scaler {} (val): {:.3f}'.format(printed_type, q))
    print('After single scaling- Avg Length {} (val): {}'.format(printed_type, after_single_scaling_avg_len))
    
    after_single_scaling_avg_cov = avg_cov(mu_calib, q * uncert_calib, target_calib)
    print('After single scaling- Avg Cov {} (val): {}'.format(printed_type, after_single_scaling_avg_cov))

    return q

def avg_len(uncert, q):
    device = uncert.device
    
    avg_len = (2 * q * uncert).mean()

    return avg_len

def avg_cov(mu, uncert, target):
    total_cov = 0.0
    for mu_single, uncert_single, target_single in zip(mu, uncert, target):
        if mu_single - uncert_single <= target_single <= mu_single + uncert_single:
            total_cov += 1.0
            
    return total_cov / len(mu)

def scale_bins_single_conformal(uncert_test, q):
    
    # Calculate Avg Length before temperature scaling
    before_scaling_avg_len = (2 * uncert_test).mean()
    print('Before scaling - Avg Length: %.3f' % (before_scaling_avg_len))
        
    # Calculate Avg Length after single scaling
    after_single_scaling_avg_len = avg_len(uncert_test, q)
    print('Optimal scaler: %.3f' % q)
    print(f'After single scaling- Avg Length: {after_single_scaling_avg_len}')
    
    return after_single_scaling_avg_len, before_scaling_avg_len



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

                            
                    
    return torch.cat(y_p_s), torch.cat(vars_s), torch.cat(logvars_s), torch.cat(targets_s)     
    
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
    alpha = 0.05
    
    print(f'alpha: {alpha}, level: {level}, base_model: {base_model}, mix_indices: {mix_indices}, save_params: {save_params}, load_params: {load_params}, calc_mean: {calc_mean}, save_test: {save_test}, load_test: {load_test}')
    
    model = BreastPathQModel(base_model, out_channels=1).to(device)

    # TODO: load checkpoint 
    # checkpoint_path = glob(f"/home/dsi/frenkel2/regression_calibration/models/{base_model}_gaussian_endovis_199_new.pth.tar")[0]
    # checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\{base_model}_gaussian_endovis_199_new.pth.tar")[0]
    
    checkpoint = torch.load(f'{models_dir}/{base_model}_gaussian_lumbar_L{level}_best.pth.tar', map_location=device)
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
    
    y_p_calib_original, vars_calib_original, logvars_calib_original, targets_calib_original = get_arrays(calib_loader, model, device)
    y_p_test_original, vars_test_original, logvars_test_original, targets_test_original = get_arrays(test_loader, model, device)
    
    
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
        target_calib = targets_calib.unsqueeze(1)

            
        err_calib = (target_calib-mu_calib).pow(2).mean(dim=1, keepdim=True).sqrt()
        uncertainty = 'aleatoric'

        uncert_calib_aleatoric = logvar_calib.exp().mean(dim=1, keepdim=True)
        uncert_calib_epistemic = var_calib.mean(dim=1, keepdim=True)

        if uncertainty == 'aleatoric':
            uncert_calib = uncert_calib_aleatoric.sqrt().clamp(0, 1)
            uncert_calib_laves = (uncert_calib_aleatoric + uncert_calib_epistemic).sqrt().clamp(0, 1)  # total
        elif uncertainty == 'epistemic':
            uncert_calib = uncert_calib_epistemic.sqrt().clamp(0, 1)
        else:
            uncert_calib = (uncert_calib_aleatoric + uncert_calib_epistemic).sqrt().clamp(0, 1)  # total
            
        
        y_p_test_list = []
        mu_test_list = []
        var_test_list = []
        logvars_test_list = []
        logvar_test_list = []
        target_test_list = []

        # test set
                                 
        y_p_test = y_p_test.clamp(0, 1).unsqueeze(1)
        mu_test = y_p_test.mean(dim=1)
        var_test = vars_test
        logvars_test = logvars_test
        logvar_test = logvars_test.mean(dim=1).unsqueeze(1)
        target_test = targets_test.unsqueeze(1)



        y_p_test_list.append(y_p_test)
        mu_test_list.append(mu_test)
        var_test_list.append(var_test)
        logvars_test_list.append(logvars_test)
        logvar_test_list.append(logvar_test)
        target_test_list.append(target_test)
                
        err_test = [(target_test-mu_test).pow(2).mean(dim=1, keepdim=True).sqrt() for target_test, mu_test in zip(target_test_list, mu_test_list)]

        uncert_aleatoric_test = [logvar_test.exp().mean(dim=1, keepdim=True) for logvar_test in logvar_test_list]
        uncert_epistemic_test = [var_test.mean(dim=1, keepdim=True) for var_test in var_test_list]

        if uncertainty == 'aleatoric':
            uncert_test = [uncert_aleatoric_t.sqrt().clamp(0, 1) for uncert_aleatoric_t in uncert_aleatoric_test]
        elif uncertainty == 'epistemic':
            uncert_test = [uncert_epistemic_t.sqrt().clamp(0, 1) for uncert_epistemic_t in uncert_epistemic_test]
        else:
            uncert_test = [(u_a_t + u_e_t).sqrt().clamp(0, 1) for u_a_t, u_e_t in zip(uncert_aleatoric_test, uncert_epistemic_test)]
                
        # CP/GC
        avg_len_before_list = []
        avg_len_single_list = []
        avg_len_single_list_gc = []

        avg_cov_before_list = []
        avg_cov_after_single_list = []
        avg_cov_after_single_list_gc = []
        
        target_calib = target_calib.mean(dim=1, keepdim=True)
        mu_calib = mu_calib.mean(dim=1, keepdim=True)
        mu_test_list = [mu_test.mean(dim=1, keepdim=True) for mu_test in mu_test_list]
        target_test_list = [target_test.mean(dim=1, keepdim=True) for target_test in target_test_list]

        for i in range(len(err_test)):
            
            q = set_scaler_conformal(target_calib, mu_calib, uncert_calib, err_calib=err_calib, gc=False, alpha=alpha)
                     
            avg_len_single, avg_len_before = scale_bins_single_conformal(uncert_test[i], q)
            
            avg_cov_before = avg_cov(mu_test_list[i], uncert_test[i], target_test_list[i])
            avg_cov_after_single = avg_cov(mu_test_list[i], q * uncert_test[i], target_test_list[i])
            
            q_gc = set_scaler_conformal(target_calib, mu_calib, uncert_calib, err_calib=err_calib, gc=True, alpha=alpha)
                     
            avg_len_single_gc, _ = scale_bins_single_conformal(uncert_test[i], q_gc)
            avg_cov_after_single_gc = avg_cov(mu_test_list[i], q_gc * uncert_test[i], target_test_list[i])
            
            
            # my methods
            
            avg_len_before_list.append(avg_len_before.cpu())
            avg_len_single_list.append(avg_len_single.cpu())
            avg_len_single_list_gc.append(avg_len_single_gc.cpu())
            
            avg_cov_before_list.append(avg_cov_before)
            avg_cov_after_single_list.append(avg_cov_after_single)
            avg_cov_after_single_list_gc.append(avg_cov_after_single_gc)
            
        print(f'Test before, Avg Length:', torch.stack(avg_len_before_list).mean().item())
        print(f'Test after single CP, Avg Length:', torch.stack(avg_len_single_list).mean().item())
        print(f'Test after single GC, Avg Length:', torch.stack(avg_len_single_list_gc).mean().item())

        print(f'Test before with Avg Cov:', torch.tensor(avg_cov_before_list).mean().item())
        print(f'Test after single CP with Avg Cov:', torch.tensor(avg_cov_after_single_list).mean().item())
        print(f'Test after single GC with Avg Cov:', torch.tensor(avg_cov_after_single_list_gc).mean().item())
        
        q_all.append(get_float(q))
        avg_len_all.append(get_float(torch.stack(avg_len_single_list).mean()))
        avg_cov_all.append(get_float(torch.tensor(avg_cov_after_single_list).mean()))
        
        q_all_gc.append(get_float(q_gc))
        avg_len_all_gc.append(get_float(torch.stack(avg_len_single_list_gc).mean()))
        avg_cov_all_gc.append(get_float(torch.tensor(avg_cov_after_single_list_gc).mean()))
        
    print(q_all)
    print(avg_len_all)
    print(avg_cov_all)
    print(q_all_gc)
    print(avg_len_all_gc)
    print(avg_cov_all_gc)

    # Define the output file path
    output_dir= '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results'
    output_file = f"lumbar_dataset_model_{base_model}_alpha_{alpha}_level_{level}_iterations_{iters}_after.txt"

    # Open the file in append mode
    with open(f'{output_dir}/{output_file}', "a") as f:
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
        
                 
        # Print and save additional info
        print(f"lumbar, {base_model}, {alpha}, {level}")
        f.write(f"lumbar, {base_model}, {alpha}, {level}\n")
    
  
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