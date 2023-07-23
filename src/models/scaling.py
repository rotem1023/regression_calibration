import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import math

from uce import uceloss


def nll_criterion_gaussian(mu, logvar, target, reduction='mean'):
    loss = torch.exp(-logvar) * torch.pow(target-mu, 2).mean(dim=1, keepdim=True) + logvar
    return loss.mean() if reduction == 'mean' else loss.sum()

def histedges_equalN(x, n_bins=15):
    npt = len(x)
    return np.interp(np.linspace(0, npt, n_bins + 1),
                    np.arange(npt),
                    np.sort(x))

def set_scaler(err_calib, uncert_calib, cross_validate='uce',
                     init_temp=2.5, log=True, num_bins=15, outlier=0.0):
    """
    Tune single scaler for the model (using the validation set) with cross-validation on NLL
    """
    # Calculate ECE before temperature scaling
    nll_criterion = nn.CrossEntropyLoss().cuda()
    before_scaling_uce = uceloss(err_calib**2, uncert_calib**2, single=True)
    if log:
        print('Before scaling - UCE: %.3f' % (before_scaling_uce * 100))
        
    # calculate optimal S
    S = (err_calib**2 / uncert_calib**2).mean().sqrt()
        
    n_bins = num_bins
    eps = 1e-5
    nll_val = 10 ** 7
    buce_val = 10 ** 7
        
    # Calculate UCE after single scaling
    after_single_scaling_uce = uceloss(err_calib**2, (S * uncert_calib)**2, single=True)
    if log:
        print('Optimal scaler: %.3f' % S)
        print('After single scaling- UCE: %.3f' % (after_single_scaling_uce * 100))
    
    init_scaler = 1.0
    
    bins_T = init_scaler*torch.ones(n_bins).cuda()
    uce_list = []
    uce_list.append(after_single_scaling_uce)
                    
    ece_ada_list = []
    count_high_acc = 0
    is_acc = False
    n, bin_boundaries = np.histogram(uncert_calib.squeeze(-1).cpu().detach(), histedges_equalN(uncert_calib.squeeze(-1).cpu().detach(), n_bins=n_bins))
    print(bin_boundaries)


    if cross_validate == 'uce':
        T_opt_buce = init_temp*torch.ones(uncert_calib.shape[0]).cuda()
        T_buce = init_temp*torch.ones(uncert_calib.shape[0]).cuda()
        buce_temperature = T_buce
    else:
        T_opt_nll = init_temp*torch.ones(uncert_calib.shape[0]).cuda()
        T_nll = init_temp*torch.ones(uncert_calib.shape[0]).cuda()
        nll_temperature = T_nll
    
    bin = 0
    
    # bin_boundaries = torch.linspace(uncert_calib.min().item(), uncert_calib.max().item(), n_bins + 1, device=device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = uncert_calib.gt(bin_lower.item()) * uncert_calib.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()  # |Bm| / n
        if prop_in_bin.item() > outlier:        
            if cross_validate == 'uce':
                errors_in_bin = (err_calib[in_bin]**2).float().mean()  # err()
                avg_uncert_in_bin = (uncert_calib[in_bin]**2).mean()  # uncert()
                S_bin = (errors_in_bin / avg_uncert_in_bin).sqrt()
                bins_T[bin] = S_bin
                avg_uncert_in_bin = ((S_bin * uncert_calib[in_bin])**2).mean()  # uncert() after calib
                buce_val = torch.abs(avg_uncert_in_bin - errors_in_bin)
            else:
                S_bin = (err_calib[in_bin]**2 / uncert_calib[in_bin]**2).mean().sqrt()
                bins_T[bin] = S_bin
                errors_in_bin = (err_calib[in_bin]**2).float().mean()  # err()
                avg_uncert_in_bin = ((S_bin * uncert_calib[in_bin])**2).mean()  # uncert()
                buce_val = torch.abs(avg_uncert_in_bin - errors_in_bin)
                                    
            samples = uncert_calib[in_bin].shape[0]
            print('uce in bin ', bin+1, ' :', (prop_in_bin * buce_val).item(), ', number of samples: ', samples)

        bin += 1

    print(bins_T)
    uncert_calib_after = uncert_calib.clone()
    for inx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = uncert_calib.gt(bin_lower.item()) * uncert_calib.le(bin_upper.item())
        uncert_calib_after[in_bin] = bins_T[inx] * uncert_calib[in_bin]
    current_uce = uceloss(err_calib**2, uncert_calib_after**2, single=True)
    print(f'After bins scaling by {cross_validate} - UCE:', current_uce.item() * 100)

    return bins_T, S, bin_boundaries, current_uce.item()

def scale_bins(err_test, uncert_test, bins_T, bin_boundaries, num_bins=15):
    uncert_test_after = uncert_test.clone()
    
    # bin_boundaries = torch.linspace(uncert_test.min().item(), uncert_test.max().item(), n_bins + 1, device=device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = uncert_test.gt(bin_lower.item()) * uncert_test.le(bin_upper.item())
        if any(in_bin):
            uncert_test_after[in_bin] = bins_T[bin] * uncert_test_after[in_bin]
        bin += 1
    uce = uceloss(err_test**2, uncert_test_after**2, single=True)
    
    return uce, uncert_test_after

def enceloss(errors, uncert, n_bins=15, outlier=0.0, range=None, single=None):
    device = errors.device
    if range == None:
        bin_boundaries = torch.linspace(uncert.min().item(), uncert.max().item(), n_bins + 1, device=device)
    else:
        bin_boundaries = torch.linspace(range[0], range[1], n_bins + 1, device=device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    errors_in_bin_list = []
    avg_uncert_in_bin_list = []
    ence_per_bin = []

    ence = torch.zeros(1, device=device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |RMV - RMSE| / RMV in each bin
        in_bin = uncert.gt(bin_lower.item()) * uncert.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean() 
        if prop_in_bin.item() > outlier:
            errors_in_bin = errors[in_bin].float().mean().sqrt()  # RMV()
            avg_uncert_in_bin = uncert[in_bin].mean().sqrt()  # RMSE()
            ence_in_bin = torch.abs(avg_uncert_in_bin - errors_in_bin) / errors_in_bin
            ence_per_bin.append(ence_in_bin)
            ence += ence_in_bin

            errors_in_bin_list.append(errors_in_bin)
            avg_uncert_in_bin_list.append(avg_uncert_in_bin)

    err_in_bin = torch.tensor(errors_in_bin_list, device=device)
    avg_uncert_in_bin = torch.tensor(avg_uncert_in_bin_list, device=device)

    if single:
        return ence.mean()
    else:
        return ence.mean(), err_in_bin, avg_uncert_in_bin, ence_per_bin
    
def calc_vars_mse_bins(errors, uncert, n_bins=15, outlier=0.0):
    device = errors.device
    n, bin_boundaries = np.histogram(uncert.squeeze(-1).cpu().detach(), histedges_equalN(uncert.squeeze(-1).cpu().detach(), n_bins=n_bins))
    # bin_boundaries = torch.linspace(uncert.min().item(), uncert.max().item(), n_bins + 1, device=device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    errors_in_bin_list = []
    uncert_in_bin_list = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = uncert.gt(bin_lower.item()) * uncert.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()  # |Bm| / n
        prop_in_bin_list.append(prop_in_bin)
        if prop_in_bin.item() > outlier:
            errors_in_bin = errors[in_bin].float()  # err()
            uncert_in_bin = uncert[in_bin]  # uncert()
            errors_in_bin_list.append(errors_in_bin)
            uncert_in_bin_list.append(uncert_in_bin)
            
    return errors_in_bin_list, uncert_in_bin_list

