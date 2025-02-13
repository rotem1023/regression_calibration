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
from data_generator_oct import OCTDataset
from models import BreastPathQModel, DistancePredictor
from glob import glob
import statistics
import math


def calc_opt_q_new_method(target_calib, mu_calib, poistive_dist, negative_dist, alpha, addtvie):
    if addtvie:
        c_min = mu_calib - negative_dist - target_calib
        c_max =  target_calib - mu_calib - poistive_dist
        c = torch.max(c_min, c_max)
    else:
        c_min  = (mu_calib - target_calib) / negative_dist
        c_max  = (target_calib - mu_calib) / poistive_dist
        c = torch.max(c_min, c_max)
    c_sorted, _ = torch.sort(c, dim=0)
    q_index = math.ceil((len(c_sorted)) * (1 - alpha))
    q = c_sorted[q_index].item()
    return q

def calc_stats_new_method(target, mu, poistive_dist, negative_dist, q, div = False):
    if div:
        lower = mu - negative_dist * q
        upper = mu + poistive_dist * q
    else:
        lower = mu - negative_dist - q
        upper = mu + poistive_dist + q
    length = calc_length(lower, upper)
    coverage = calc_coverage(lower, upper, target)
    return length, coverage

def  calc_coverage(lower, upper, target):
    coverage = (lower <= target) & (target <= upper)
    return coverage.float().mean().item()

def calc_length(lower, upper):
    return torch.mean(abs(upper - lower)).item()

def get_arrays(data_loader):
    data_s = []
    targets_s = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data, target
            data_s.append(data)
            targets_s.append(target.detach())  

            
                             
    return torch.cat(data_s).cpu(), torch.cat(targets_s).cpu()


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

def main():
    eval_single_img = False
    data_dir = "C:\lior\studies\master\projects\calibration/regression calibration/3doct-pose-dataset/data/"
    
    mix_indices = True
    save_params = True
    load_params = False
    save_test = True
    load_test = False
    calc_mean = True
    eval_test_set(save_params=save_params, mix_indices=mix_indices, load_params=load_params, calc_mean=calc_mean, save_test=save_test, load_test=load_test)    
    
def eval_test_set(save_params=False, load_params=False, mix_indices=True, calc_mean=False, save_test=False, load_test=False):
    base_model = 'densenet201'
    assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
    device = torch.device("cuda:3")
    
    alpha = 0.05
    
    model = BreastPathQModel(base_model, out_channels=6).to(device)

    # checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\{base_model}_gaussian_oct_315.pth.tar")[0] # efficientnet
    checkpoint_path = f"/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots/{base_model}_gaussian_oct_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("Loading previous weights at epoch " + str(checkpoint['epoch']) + " from\n" + checkpoint_path)
    
    dist_model_name = 'resnet50'
    dist_model = DistancePredictor(dist_model_name, in_channels=3).to(device)
    checkpoint = torch.load(f'/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots_new/{dist_model_name}_oct_snapshot_dist_{base_model}_lambda_1_scale_factor1_new.pth.tar', map_location=device)
    dist_model.load_state_dict(checkpoint['state_dict'])
    dist_model.eval()
    
    batch_size = 16
    resize_to = (256, 256)

    
    q_all = []
    avg_len_all = []
    avg_cov_all = []
    q_all_gc = []
    avg_len_all_gc = []
    avg_cov_all_gc = []
    q_all_new_method = []
    avg_len_all_new_method = []
    avg_cov_all_new_method = []
    
    
    
    data_set_valid_original = OCTDataset(group='valid')
    data_set_test_original = OCTDataset(group='test')
    calib_loader = torch.utils.data.DataLoader(data_set_valid_original, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(data_set_test_original, batch_size=batch_size, shuffle=False)
    data_calib_original, target_calib_original = get_arrays(calib_loader)
    data_test_original, target_test_original = get_arrays(test_loader)

    for _ in range(20):
        calib_arrays = [data_calib_original, target_calib_original]
        test_arrays = [data_test_original, target_test_original]
            
        calib_shuffled, test_shuffled = shuffle_arrays(calib_arrays, test_arrays)
            
        data_calib, target_calib = calib_shuffled
        data_test, target_test = test_shuffled
        calib_dataset = torch.utils.data.TensorDataset(data_calib, target_calib)
        test_dataset = torch.utils.data.TensorDataset(data_test, target_test)
            
        calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        model.eval()
        y_p_calib = []
        vars_calib = []
        logvars_calib = []
        targets_calib = []
        distance_minus_calib = []
        distance_plus_calib = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(calib_loader)):
                data, target = data.to(device), target.to(device)

                y_p, logvar, var_bayesian = model(data, dropout=True, mc_dropout=True, test=True)

                y_p_calib.append(y_p.detach())
                vars_calib.append(var_bayesian.detach())
                logvars_calib.append(logvar.detach())
                targets_calib.append(target.detach())
                
                distances = dist_model(data).detach()
                distance_plus_calib.append(distances[:,0])
                distance_minus_calib.append(distances[:,1])
        
        
        y_p_calib = torch.cat(y_p_calib, dim=1).clamp(0, 1).permute(1,0,2)
        mu_calib = y_p_calib.mean(dim=1)
        var_calib = torch.cat(vars_calib, dim=0)
        logvars_calib = torch.cat(logvars_calib, dim=1).permute(1,0,2)
        logvar_calib = logvars_calib.mean(dim=1)
        target_calib = torch.cat(targets_calib, dim=0)
        distance_plus_calib = torch.cat(distance_plus_calib, dim = 0).unsqueeze(-1)
        distance_minus_calib = torch.cat(distance_minus_calib, dim = 0).unsqueeze(-1)
        
        err_calib = (target_calib-mu_calib).pow(2).mean(dim=1, keepdim=True).sqrt()

        uncertainty = 'aleatoric'

        uncert_calib_aleatoric = logvar_calib.exp().mean(dim=1, keepdim=True)
        uncert_calib_epistemic = var_calib.mean(dim=1, keepdim=True)

        if uncertainty == 'aleatoric':
            uncert_calib = uncert_calib_aleatoric.sqrt().clamp(0, 1)
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

        for i in range(1):
            y_p_test = []
            mus_test = []
            vars_test = []
            logvars_test = []
            targets_test = []
            distance_minus_test = []
            distance_plus_test = []

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
                    data, target = data.to(device), target.to(device)

                    y_p, logvar, var_bayesian = model(data, dropout=True, mc_dropout=True, test=True)

                    y_p_test.append(y_p.detach())
                    vars_test.append(var_bayesian.detach())
                    logvars_test.append(logvar.detach())
                    targets_test.append(target.detach())
                    distances = dist_model(data).detach()
                    distance_plus_test.append(distances[:,0])
                    distance_minus_test.append(distances[:,1])

                y_p_test = torch.cat(y_p_test, dim=1).clamp(0, 1).permute(1,0,2)
                mu_test = y_p_test.mean(dim=1)
                var_test = torch.cat(vars_test, dim=0)
                logvars_test = torch.cat(logvars_test, dim=1).permute(1,0,2)
                logvar_test = logvars_test.mean(dim=1)
                target_test = torch.cat(targets_test, dim=0)
                distance_plus_test = torch.cat(distance_plus_test, dim = 0).unsqueeze(-1)
                distance_minus_test = torch.cat(distance_minus_test, dim = 0).unsqueeze(-1)

                y_p_test_list.append(y_p_test)
                mu_test_list.append(mu_test)
                var_test_list.append(var_test)
                logvars_test_list.append(logvars_test)
                logvar_test_list.append(logvar_test)
                target_test_list.append(target_test)
                
        err_test = [(target_test-mu_test).pow(2).mean(dim=1, keepdim=True).sqrt() for target_test, mu_test in zip(target_test_list, mu_test_list)]
        errvar_test = [(y_p_test-target_test.unsqueeze(1).repeat(1,25,1)).pow(2).mean(dim=(1,2)).unsqueeze(-1) for target_test, y_p_test in zip(target_test_list, y_p_test_list)]

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
            
            avg_len_before_list.append(avg_len_before.cpu())
            avg_len_single_list.append(avg_len_single.cpu())
            avg_len_single_list_gc.append(avg_len_single_gc.cpu())
            
            avg_cov_before_list.append(avg_cov_before)
            avg_cov_after_single_list.append(avg_cov_after_single)
            avg_cov_after_single_list_gc.append(avg_cov_after_single_gc)
            
            
            q_add = calc_opt_q_new_method(target_calib, mu_calib, distance_plus_calib, distance_minus_calib, alpha , True)
            length_add_calib, coverage_add_calib = calc_stats_new_method(target_calib, mu_calib, distance_plus_calib, distance_minus_calib, q_add, div=False)    
            print(f'q_add: {q_add}, avg_len_single_new_add_val: {length_add_calib}, avg_cov_after_single_new_add_val: {coverage_add_calib}')
            
            # cal avg new len and cov test set
        
            length_add_test, coverage_add_test = calc_stats_new_method(target_test_list[i], mu_test_list[i], distance_plus_test, distance_minus_test, q_add, div=False)
            print(f'q_add: {q_add}, avg_len_single_new_add_test: {length_add_test}, avg_cov_after_single_new_add_test: {coverage_add_test}')
            
            q_all_new_method.append(q_add)
            avg_len_all_new_method.append(length_add_test)
            avg_cov_all_new_method.append(coverage_add_test)
            
        if calc_mean:
            top_limit = mu_test_list[0] + uncert_test[0] * q
            bottom_limit = mu_test_list[0] - uncert_test[0] * q
            
            pred = ((top_limit + bottom_limit) / 2).squeeze(1)
            mse_cp = torch.nn.functional.mse_loss(pred, target_test_list[0].mean(dim=1))
            
            top_limit_gc = mu_test_list[0] + uncert_test[0] * q_gc
            bottom_limit_gc = mu_test_list[0] - uncert_test[0] * q_gc
            
            pred_gc = ((top_limit_gc + bottom_limit_gc) / 2).squeeze(1)
            mse_gc = torch.nn.functional.mse_loss(pred_gc, target_test_list[0].mean(dim=1))
            
        print(f'Test before, Avg Length:', torch.stack(avg_len_before_list).mean().item())
        print(f'Test after single CP, Avg Length:', torch.stack(avg_len_single_list).mean().item())
        print(f'Test after single GC, Avg Length:', torch.stack(avg_len_single_list_gc).mean().item())

        print(f'Test before with Avg Cov:', torch.tensor(avg_cov_before_list).mean().item())
        print(f'Test after single CP with Avg Cov:', torch.tensor(avg_cov_after_single_list).mean().item())
        print(f'Test after single GC with Avg Cov:', torch.tensor(avg_cov_after_single_list_gc).mean().item())
        
        print(f'Test MSE CP:', mse_cp.item())
        print(f'Test MSE GC:', mse_gc.item())
        
        q_all.append(q)
        avg_len_all.append(torch.stack(avg_len_single_list).mean().item())
        avg_cov_all.append(torch.tensor(avg_cov_after_single_list).mean().item())
        
        q_all_gc.append(q_gc)
        avg_len_all_gc.append(torch.stack(avg_len_single_list_gc).mean().item())
        avg_cov_all_gc.append(torch.tensor(avg_cov_after_single_list_gc).mean().item())
        
    print(f"final results")
    print("cp")   
    print(q_all)
    print(avg_len_all)
    print(avg_cov_all)
    print("gc")
    print(q_all_gc)
    print(avg_len_all_gc)
    print(avg_cov_all_gc)
    print("new method")
    print(q_all_new_method)
    print(avg_len_all_new_method)
    print(avg_cov_all_new_method)


    print(f'q CP mean: {statistics.mean(q_all)}, q CP std: {statistics.stdev(q_all)}')
    print(f'avg_len CP mean: {statistics.mean(avg_len_all)}, avg_len CP std: {statistics.stdev(avg_len_all)}')
    print(f'avg_cov CP mean: {statistics.mean(avg_cov_all)}, avg_cov CP std: {statistics.stdev(avg_cov_all)}')
        
    print(f'q GC mean: {statistics.mean(q_all_gc)}, q GC std: {statistics.stdev(q_all_gc)}')
    print(f'avg_len GC mean: {statistics.mean(avg_len_all_gc)}, avg_len GC std: {statistics.stdev(avg_len_all_gc)}')
    print(f'avg_cov GC mean: {statistics.mean(avg_cov_all_gc)}, avg_cov GC std: {statistics.stdev(avg_cov_all_gc)}')

    print(f'q new method mean: {statistics.mean(q_all_new_method)}, q GC std: {statistics.stdev(q_all_new_method)}')
    print(f'avg_len method mean: {statistics.mean(avg_len_all_new_method)}, avg_len GC std: {statistics.stdev(avg_len_all_new_method)}')
    print(f'avg_cov method mean: {statistics.mean(avg_cov_all_new_method)}, avg_cov GC std: {statistics.stdev(avg_cov_all_new_method)}')

    print(f"oct, {base_model}, {alpha}")
    
    
if __name__ == '__main__':
    main()