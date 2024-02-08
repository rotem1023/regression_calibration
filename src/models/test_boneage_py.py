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
from data_generator_boneage import BoneAgeDataset
from models import BreastPathQModel
from glob import glob
import statistics
import math
import pandas as pd
from skimage import io
import pickle


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
    data_dir = "C:\lior\studies\master\projects\calibration/regression calibration/rsna-bone-age"
    
    if eval_single_img:
        image_name = '11613' # 7246 / 6956 / 11613
        image_path = f"C:\lior\studies\master\projects\calibration/regression calibration/rsna-bone-age/boneage-training-dataset/{image_name}.png"
        eval_single(data_dir, image_path)
    else:
        mix_indices = False
        save_params = False
        load_params = True
        save_test = False
        load_test = True
        calc_mean = True
        eval_test_set(data_dir, save_params=save_params, mix_indices=mix_indices, load_params=load_params, calc_mean=calc_mean, save_test=save_test, load_test=load_test)
        
        
def eval_test_set(data_dir="C:\lior\studies\master\projects\calibration/regression calibration/rsna-bone-age", save_params=False, load_params=False, mix_indices=True, calc_mean=False, save_test=False, load_test=False):
    base_model = 'efficientnetb4'
    assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
    device = torch.device("cuda:0")
    
    alpha = 0.05
    
    model = BreastPathQModel(base_model, in_channels=1, out_channels=1).to(device)

    checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\{base_model}_gaussian_boneage_499.pth.tar")[0]

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("Loading previous weights at epoch " + str(checkpoint['epoch']) + " from\n" + checkpoint_path)
    
    batch_size = 16
    resize_to = (256, 256)

    # data_dir = '/home/dsi/frenkel2/data/rsna-bone-age/'
    data_set = BoneAgeDataset(data_dir=data_dir, augment=False, resize_to=resize_to)
    assert len(data_set) > 0

    calib_indices = torch.load('./data_indices/boneage_valid_indices.pth')
    test_indices = torch.load('./data_indices/boneage_test_indices.pth')

    print(calib_indices.shape)
    print(test_indices.shape)
    
    calib_original_indices = calib_indices.clone()
    test_original_indices = test_indices.clone()
    
    q_all = []
    avg_len_all = []
    avg_cov_all = []
    q_all_gc = []
    avg_len_all_gc = []
    avg_cov_all_gc = []

    for _ in range(1):
        if mix_indices:
            val_original_shape = calib_original_indices.shape[0]
            test_original_shape = test_original_indices.shape[0]

            all_indices = torch.cat((calib_original_indices, test_original_indices))
            idx = torch.randperm(all_indices.nelement())
            all_indices = all_indices.view(-1)[idx].view(all_indices.size())

            calib_indices, test_indices = torch.split(all_indices, [val_original_shape, test_original_shape])

            print(calib_indices.shape)
            print(test_indices.shape)

        calib_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size,
                                                sampler=SubsetRandomSampler(calib_indices))
        test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size,
                                                sampler=SubsetRandomSampler(test_indices))
        
        model.eval()
        y_p_calib = []
        vars_calib = []
        logvars_calib = []
        targets_calib = []
        
        if load_params:
            load_path = 'C:/lior/studies/master/projects/calibration/regression calibration/regression_calibration/reports/var_and_mse_calib/'
            with open(load_path + f'{base_model}_gaussian_boneage_calib_params.pickle', 'rb') as handle:
                calib_dict = pickle.load(handle)
                mu_calib = calib_dict['mu']
                target_calib = calib_dict['target']
                err_calib = calib_dict['err']
                uncert_calib = calib_dict['uncert']
                
        else:

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(tqdm(calib_loader)):
                    data, target = data.to(device), target.to(device)

                    y_p, logvar, var_bayesian = model(data, dropout=True, mc_dropout=True, test=True)

                    y_p_calib.append(y_p.detach())
                    vars_calib.append(var_bayesian.detach())
                    logvars_calib.append(logvar.detach())
                    targets_calib.append(target.detach())
            
            
                y_p_calib = torch.cat(y_p_calib, dim=1).clamp(0, 1).permute(1,0,2)
                mu_calib = y_p_calib.mean(dim=1)
                var_calib = torch.cat(vars_calib, dim=0)
                logvars_calib = torch.cat(logvars_calib, dim=1).permute(1,0,2)
                logvar_calib = logvars_calib.mean(dim=1)
                target_calib = torch.cat(targets_calib, dim=0)
                
                err_calib = (target_calib-mu_calib).pow(2).mean(dim=1, keepdim=True).sqrt()
                errvar_calib = (y_p_calib-target_calib.unsqueeze(1).repeat(1,25,1)).pow(2).mean(dim=(1,2)).unsqueeze(-1)

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
                
            if save_params:
                save_path = 'C:/lior/studies/master/projects/calibration/regression calibration/regression_calibration/reports/var_and_mse_calib/'
                with open(save_path + f'{base_model}_gaussian_boneage_calib_params.pickle', 'wb') as handle:
                    pickle.dump({'mu': mu_calib,
                                'target': target_calib,
                                'err': err_calib, 
                                'uncert': uncert_calib,
                                }
                                , handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        if load_test:
            load_path = 'C:/lior/studies/master/projects/calibration/regression calibration/regression_calibration/reports/var_and_mse_calib/'
            with open(load_path + f'{base_model}_gaussian_boneage_test_results.pickle', 'rb') as handle:
                calib_dict = pickle.load(handle)
                y_p_test_list = calib_dict['yp']
                mu_test_list = calib_dict['mu']
                logvar_test_list = calib_dict['logvar']
                target_test_list = calib_dict['target']
                
        else:
        
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

                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
                        data, target = data.to(device), target.to(device)

                        y_p, logvar, var_bayesian = model(data, dropout=True, mc_dropout=True, test=True)

                        y_p_test.append(y_p.detach())
                        vars_test.append(var_bayesian.detach())
                        logvars_test.append(logvar.detach())
                        targets_test.append(target.detach())

                    y_p_test = torch.cat(y_p_test, dim=1).clamp(0, 1).permute(1,0,2)
                    mu_test = y_p_test.mean(dim=1)
                    var_test = torch.cat(vars_test, dim=0)
                    logvars_test = torch.cat(logvars_test, dim=1).permute(1,0,2)
                    logvar_test = logvars_test.mean(dim=1)
                    target_test = torch.cat(targets_test, dim=0)

                    y_p_test_list.append(y_p_test)
                    mu_test_list.append(mu_test)
                    var_test_list.append(var_test)
                    logvars_test_list.append(logvars_test)
                    logvar_test_list.append(logvar_test)
                    target_test_list.append(target_test)
                    
            if save_test:
                save_path = 'C:/lior/studies/master/projects/calibration/regression calibration/regression_calibration/reports/var_and_mse_calib/'
                with open(save_path + f'{base_model}_gaussian_boneage_test_results.pickle', 'wb') as handle:
                    pickle.dump({'yp': y_p_test_list, 'mu': mu_test_list, 'target': target_test_list, 'logvar': logvar_test_list}, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
        err_test = [(target_test-mu_test).pow(2).mean(dim=1, keepdim=True).sqrt() for target_test, mu_test in zip(target_test_list, mu_test_list)]
        errvar_test = [(y_p_test-target_test.unsqueeze(1).repeat(1,25,1)).pow(2).mean(dim=(1,2)).unsqueeze(-1) for target_test, y_p_test in zip(target_test_list, y_p_test_list)]

        uncert_aleatoric_test = [logvar_test.exp().mean(dim=1, keepdim=True) for logvar_test in logvar_test_list]
        # uncert_epistemic_test = [var_test.mean(dim=1, keepdim=True) for var_test in var_test_list]
        
        uncertainty = 'aleatoric'

        if uncertainty == 'aleatoric':
            uncert_test = [uncert_aleatoric_t.sqrt().clamp(0, 1) for uncert_aleatoric_t in uncert_aleatoric_test]
            # uncert_test_laves = [(u_a_t + u_e_t).sqrt().clamp(0, 1) for u_a_t, u_e_t in zip(uncert_aleatoric_test, uncert_epistemic_test)]
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
        
    print(q_all)
    print(avg_len_all)
    print(avg_cov_all)
    print(q_all_gc)
    print(avg_len_all_gc)
    print(avg_cov_all_gc)

    if len(q_all) > 1:
        print(f'q CP mean: {statistics.mean(q_all)}, q CP std: {statistics.stdev(q_all)}')
        print(f'avg_len CP mean: {statistics.mean(avg_len_all)}, avg_len CP std: {statistics.stdev(avg_len_all)}')
        print(f'avg_cov CP mean: {statistics.mean(avg_cov_all)}, avg_cov CP std: {statistics.stdev(avg_cov_all)}')
        
        print(f'q GC mean: {statistics.mean(q_all_gc)}, q GC std: {statistics.stdev(q_all_gc)}')
        print(f'avg_len GC mean: {statistics.mean(avg_len_all_gc)}, avg_len GC std: {statistics.stdev(avg_len_all_gc)}')
        print(f'avg_cov GC mean: {statistics.mean(avg_cov_all_gc)}, avg_cov GC std: {statistics.stdev(avg_cov_all_gc)}')

    
def eval_single(data_dir, image_path, labels_min=1.0, labels_max=228.0):
    base_model = 'densenet201'
    assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
    device = torch.device("cuda:0")
    
    alpha = 0.1
    
    model = BreastPathQModel(base_model, in_channels=1, out_channels=1).to(device)

    checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\{base_model}_gaussian_boneage_493.pth.tar")[0]

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("Loading previous weights at epoch " + str(checkpoint['epoch']) + " from\n" + checkpoint_path)
    model.eval()
    
    batch_size = 16
    resize_to = (256, 256)
    
    x = io.imread(image_path, as_gray=True)
    x = np.atleast_3d(x)
    max_size = np.max(x.shape)

    trans_always1 = [
        transforms.ToPILImage(),
        transforms.CenterCrop(max_size),
        transforms.Resize(resize_to),
    ]

    trans = transforms.Compose(trans_always1)
    x = trans(x)
    w, h = x.size
    size = (h, w)
    
    image_name = int(image_path.split('/')[-1].split('.')[0])
    df = pd.read_csv(data_dir+f'/boneage-training-dataset.csv')
    label = df[df['id'] == image_name]['boneage']
    
    label = np.array(label, dtype=np.float64)
    label = label - labels_min
    label = label / labels_max
    label = torch.tensor(label).float().unsqueeze(-1).unsqueeze(0)
    
    mean = [0.14344494]
    std = [0.18635063]

    trans_always2 = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    trans_augment = []
    trans = transforms.Compose(trans_augment+trans_always2)

    x = trans(x).unsqueeze(0)
    
    with torch.no_grad():
        data, target = x.to(device), label.to(device)

        y_p, logvar, var_bayesian = model(data, dropout=True, mc_dropout=True, test=True)
        
        y_p_test = y_p.detach().clamp(0, 1).permute(1,0,2)
        mu_test = y_p_test.mean(dim=1)
        logvar_test = logvar.permute(1,0,2).mean(dim=1)
        uncert_test = logvar_test.exp().mean(dim=1, keepdim=True).sqrt().clamp(0, 1)
        
    load_path = 'C:/lior/studies/master/projects/calibration/regression calibration/regression_calibration/reports/var_and_mse_calib/'
    with open(load_path + f'{base_model}_gaussian_boneage_calib_params.pickle', 'rb') as handle:
        calib_dict = pickle.load(handle)
        uncert_calib = calib_dict['uncert']
        mu_calib = calib_dict['mu']
        target_calib = calib_dict['target']
        err_calib = calib_dict['err']

    q = set_scaler_conformal(target_calib, mu_calib, uncert_calib, err_calib=err_calib, gc=False, alpha=alpha)
    
    top_limit = mu_test + q * uncert_test
    bottom_limit = mu_test - q * uncert_test
    
    q_gc = set_scaler_conformal(target_calib, mu_calib, uncert_calib, err_calib=err_calib, gc=True, alpha=alpha)
    
    top_limit_gc = mu_test + q_gc * uncert_test
    bottom_limit_gc = mu_test - q_gc * uncert_test
    
    print(f'Test target for {image_name}: {target.item()}\n')
    
    print(f'Test top limit CP for {image_name}:', top_limit.item())
    print(f'Test top limit CP for {image_name}:',bottom_limit.item())

    print(f'Test top limit GC for {image_name}:', top_limit_gc.item())
    print(f'Test top limit GC for {image_name}:',bottom_limit_gc.item())
    
    
if __name__ == '__main__':
    main()