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
from data_generator_oct import OCTDataset
from functools import reduce
from operator import mul
from max_rank import adjusted_q_max_rank
    
    

def calc_optimal_q_max_rank(target, preds, alpha):
    dims = preds.shape[0]
    scores = []
    for i in range(dims):
        dim_preds = preds[i]
        dim_target = target[:, i]
        y_lower = dim_preds[:, 0]
        y_upper = dim_preds[:, 1]
        error_low = y_lower - dim_target
        error_high = dim_target - y_upper
        err = torch.maximum(error_high, error_low)
        scores.append(err)
    scores = torch.stack(scores, dim=0)
    return adjusted_q_max_rank(scores.T.cpu().numpy(), alpha=alpha) 
        
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
    length = torch.clamp(abs(y_upper - y_lower),max=1)
    return ((y_test >= y_lower) & (y_test <= y_upper)), length

def compute_coverage_len(targets, preds, q_s):
    coverages, lengths = [], []
    ndim = targets.shape[1]
    for i in range(ndim):
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

def create_final_preds_oct(t_p_s):
    preds = create_final_preds(t_p_s)
    return preds.permute(2,1,0)
   

class DataToVisualize:
    def __init__(self, x, lower_preds_x, lower_preds_y, higer_preds_x, higher_preds_y, target):
        """
        Initialize the DataToVisualize object with input data, predictions, and standard deviations.
        
        Args:
            x (Tensor): Input data tensor.
            mu (Tensor): Predicted mean tensor.
            sd (Tensor): Standard deviation tensor.
        """
        self.x = x
        self.lower_preds_x = lower_preds_x
        self.lower_preds_y = lower_preds_y
        self.higher_preds_x = higer_preds_x
        self.higher_preds_y = higher_preds_y
        self.target = target

def create_data_to_visualize(data,preds, targets):
    visaul_data_dim = data.shape[0]
    preds_x = preds[0]
    preds_y = preds[1]
    lower_preds_x = preds_x[:visaul_data_dim, 0]
    upper_preds_x = preds_x[:visaul_data_dim, 1]
    lower_preds_y = preds_y[:visaul_data_dim, 0]
    upper_preds_y = preds_y[:visaul_data_dim, 1]
    targets = targets[:visaul_data_dim]
    
    return DataToVisualize(data, lower_preds_x=lower_preds_x,
                           lower_preds_y=lower_preds_y,
                           higer_preds_x=upper_preds_x,
                           higher_preds_y=upper_preds_y,
                           target=targets) 


def get_arrays(data_loader, model, device, dataset):
    t_p_s = []
    targets_s = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)

            t_p = model(data, dropout=True, mc_dropout=True, test=True)

            t_p_s.append(t_p)

            targets_s.append(target.detach()) 
            if batch_idx ==0:
                data_to_visualize = data.cpu().numpy()
                break


        targets = torch.cat(targets_s).cpu()
        # if dataset == 'oct':
        final_preds = create_final_preds_oct(t_p_s)
        # else:
        #     final_preds = create_final_preds(t_p_s)
        
                                
    return final_preds, targets, create_data_to_visualize(data_to_visualize, final_preds, targets) 
    
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
    device = torch.device("cuda:1")
    dataset = 'lumbar'
    iters = 20
    level = 1
    alpha = 0.05
    load_preds = False
    save_visual = True
    
    print(f'Running CQR for model {base_model} with alpha {alpha} and level {level}, {iters} iterations')
    
    
    results_dir = "/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results/predictions/cqr/dims"
    if not load_preds:
        model = BreastPathQModel(base_model, out_channels=2).to(device)

        # checkpoint_path = glob(f"/home/dsi/frenkel2/regression_calibration/models/{base_model}_gaussian_endovis_199_new.pth.tar")[0]
        # checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\{base_model}_gaussian_endovis_199_new.pth.tar")[0]
        if dataset == 'oct':
            out_channels = 6
            checkpoint = torch.load(f'{models_dir}/{base_model}_{dataset}_L1_alpha_{alpha}_cqr_dims_best.pth.tar', map_location=device)
        else:
            out_channels = 2
            checkpoint = torch.load(f'{models_dir}/{base_model}_lumbar_L{level}_alpha_{alpha}_cqr_dims_best.pth.tar', map_location=device)
        model = BreastPathQModel(base_model, out_channels=out_channels).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"epoch: {checkpoint['epoch']}")
        model.eval()
        batch_size = 64

        if dataset =='brain':
            data_set_valid_original = BrainDatasetVal(model=base_model)
            data_set_test_original = BrainDatasetTest(model=base_model)
        elif dataset == 'oct':
            in_channels = 3
            out_channels = 6
            resize_to = (256, 256)

            data_set_valid_original = OCTDataset(group='train', augment=False, resize_to=resize_to)
            data_set_test_original = OCTDataset(group='valid', augment=False, resize_to=resize_to)
        else:
            data_set_valid_original = LumbarDataset(level=level, mode='val', augment=False, scale=0.5)
            data_set_test_original = LumbarDataset(level=level, mode='test', augment=False, scale=0.5)
        
        assert len(data_set_valid_original) > 0
        assert len(data_set_test_original) > 0
        print(len(data_set_valid_original))
        print(len(data_set_test_original))
            
        calib_loader = torch.utils.data.DataLoader(data_set_valid_original, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(data_set_test_original, batch_size=batch_size, shuffle=False)
        
        y_p_calib_original, targets_calib_original, test_visual = get_arrays(calib_loader, model, device, dataset = dataset)
        y_p_test_original, targets_test_original, valid_visual = get_arrays(test_loader, model, device, dataset  = dataset)
        
        # save arrays
        os.makedirs(results_dir, exist_ok=True)
        
        if save_visual:
            torch.save({'x': test_visual.x,
                        'lower_preds_x': test_visual.lower_preds_x,
                        'lower_preds_y': test_visual.lower_preds_y,
                        'higher_preds_x': test_visual.higher_preds_x,
                        'higher_preds_y': test_visual.higher_preds_y,
                        'target': test_visual.target}, f'{results_dir}/{dataset}_dataset_cqr_model_{base_model}_alpha_{alpha}_level_{level}_test_visual.pth.tar')
        
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
    q_all_max_rank= []
    
    len_valid_sets = []
    cov_valid_sets = []
    len_valid_sets_max_rank = []
    cov_valid_sets_max_rank = []

    len_test_sets = []
    cov_test_sets = []
    len_test_sets_max_rank = []
    cov_test_sets_max_rank = []    


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
        length_val, coverage_val = compute_coverage_len(target_calib, t_p_calib, q)
        length_test, coverage_test = compute_coverage_len(target_test, t_p_test, q)  
        
        q_max_rank = calc_optimal_q_max_rank(target_calib, t_p_calib, alpha)
        valid_length_max_rank, valid_coverage_max_rank = compute_coverage_len(target_calib, t_p_calib, q_max_rank)
        test_length_max_rank, test_coverage_max_rank = compute_coverage_len(target_test, t_p_test, q_max_rank)
          
    
        
        q_all.append(get_float(q))
        len_valid_sets.append(get_float(length_val))
        cov_valid_sets.append(get_float(coverage_val))
        len_test_sets.append(get_float(length_test))
        cov_test_sets.append(get_float(coverage_test))
        
        q_all_max_rank.append(get_float(q_max_rank))
        len_valid_sets_max_rank.append(get_float(valid_length_max_rank))
        cov_valid_sets_max_rank.append(get_float(valid_coverage_max_rank))
        len_test_sets_max_rank.append(get_float(test_length_max_rank))
        cov_test_sets_max_rank.append(get_float(test_coverage_max_rank))
        
    print(f" q's: {q_all}")
    print(f"valid len's {len_valid_sets}")
    print(f"valid coverage's {cov_valid_sets}")
    print(f"test coverages's {len_test_sets}")
    print(f"test coverage's {cov_test_sets}")
    print(f" q's max rank: {q_all_max_rank}")
    print(f"valid len's max rank {len_valid_sets_max_rank}")
    print(f"valid coverage's max rank {cov_valid_sets_max_rank}")
    print(f"test len's max rank {len_test_sets_max_rank}")
    print(f"test coverage's max rank {cov_test_sets_max_rank}")

    # Define the output file path
    output_dir= '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results/cqr/dims'
    output_file = f"{dataset}_dataset_model_{base_model}_alpha_{alpha}_level_{level}_iterations_{iters}.txt"

    # Open the file in append mode
    with open(f'{output_dir}/{output_file}', "w") as f:
        # Print and save CP metrics
        print(f'q cqr mean: {np.array(q_all).mean(axis=0).tolist()}, q Bonf std: {np.array(q_all).std(axis=0).tolist()}')
        f.write(f'q cqr mean: {np.array(q_all).mean(axis=0).tolist()}, q Bonf std: {np.array(q_all).std(axis=0).tolist()}\n')
        
        
        print(f'avg_len valid mean: {statistics.mean(len_valid_sets)}, avg_len valid std: {statistics.stdev(len_valid_sets)}')
        f.write(f'avg_len valid  mean: {statistics.mean(len_valid_sets)}, avg_len valid std: {statistics.stdev(len_valid_sets)}\n')
        
        print(f'avg_cov valid mean: {statistics.mean(cov_valid_sets)}, avg_cov std: {statistics.stdev(cov_valid_sets)}')
        f.write(f'avg_cov valid mean: {statistics.mean(cov_valid_sets)}, avg_cov std: {statistics.stdev(cov_valid_sets)}\n')
        
        
        print(f'q max_rank mean: {np.array(q_all_max_rank).mean(axis=0).tolist()}, q max_rank std: {np.array(q_all_max_rank).std(axis=0).tolist()}')
        f.write(f'q max_rank mean: {np.array(q_all_max_rank).mean(axis=0).tolist()}, q max_rank std: {np.array(q_all_max_rank).std(axis=0).tolist()}\n')
        
        print(f'avg_len valid max_rank mean: {statistics.mean(len_valid_sets_max_rank)}, avg_len valid max_rank std: {statistics.stdev(len_valid_sets_max_rank)}')
        f.write(f'avg_len valid max_rank mean: {statistics.mean(len_valid_sets_max_rank)}, avg_len valid max_rank std: {statistics.stdev(len_valid_sets_max_rank)}\n')
        
        print(f'avg_cov valid max_rank mean: {statistics.mean(cov_valid_sets_max_rank)}, avg_cov valid max_rank std: {statistics.stdev(cov_valid_sets_max_rank)}')
        f.write(f'avg_cov valid max_rank mean: {statistics.mean(cov_valid_sets_max_rank)}, avg_cov valid max_rank std: {statistics.stdev(cov_valid_sets_max_rank)}\n')
        
        print(f'avg_len test mean: {statistics.mean(len_test_sets)}, avg_len  test std: {statistics.stdev(len_test_sets)}')
        f.write(f'avg_len test mean: {statistics.mean(len_test_sets)}, avg_len test std: {statistics.stdev(len_test_sets)}\n')
        
        print(f'avg_cov test mean: {statistics.mean(cov_test_sets)}, avg_cov test std: {statistics.stdev(cov_test_sets)}')
        f.write(f'avg_cov test mean: {statistics.mean(cov_test_sets)}, avg_cov test std: {statistics.stdev(cov_test_sets)}\n')
        
        print(f'avg_len test max_rank mean: {statistics.mean(len_test_sets_max_rank)}, avg_len test max_rank std: {statistics.stdev(len_test_sets_max_rank)}')
        f.write(f'avg_len test max_rank mean: {statistics.mean(len_test_sets_max_rank)}, avg_len test max_rank std: {statistics.stdev(len_test_sets_max_rank)}\n')
        
        print(f'avg_cov test max_rank mean: {statistics.mean(cov_test_sets_max_rank)}, avg_cov test max_rank std: {statistics.stdev(cov_test_sets_max_rank)}')
        f.write(f'avg_cov test max_rank mean: {statistics.mean(cov_test_sets_max_rank)}, avg_cov test max_rank std: {statistics.stdev(cov_test_sets_max_rank)}\n')
                 
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