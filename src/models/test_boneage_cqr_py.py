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
from cqr_model import BreastPathQModel
from models import BreastPathQModel as BreastPathQModelGauss
from glob import glob
import statistics
import math
from skimage import io
import pickle
import pandas as pd


use_gauss_model = True


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
    
    return q

    #CQR prediction

def set_scaler_conformal(target_calib, mu_calib, init_temp=2.5, log=True, alpha=0.1):
    """
    Tune single scaler for the model (using the validation set) with cross-validation on NLL
    """
        
    printed_type = 'CQR'
            
    # Calculate optimal q
    q = calc_optimal_q(target_calib.mean(dim=1), mu_calib, alpha=alpha)
    
    after_single_scaling_avg_len = torch.mean(abs((mu_calib[:, 1] + q) - (mu_calib[:, 0] - q)))
    # after_single_scaling_avg_len = ((mu_calib[1] + q) - (mu_calib[0] - q)).mean()
    print('Optimal scaler {} (val): {:.3f}'.format(printed_type, q))
    print('After single scaling- Avg Length {} (val): {}'.format(printed_type, after_single_scaling_avg_len))
    
    after_single_scaling_avg_cov = avg_cov(mu_calib, q, target_calib.mean(dim=1))
    print('After single scaling- Avg Cov {} (val): {}'.format(printed_type, after_single_scaling_avg_cov))
    
    return q

def avg_cov(mu, q, target, before=False):
    if before:
        in_the_range = torch.sum((target >= mu[:, 0]) & (target <= mu[:, 1]))
    else:
        in_the_range = torch.sum((target >= (mu[:, 0] - q)) & (target <= (mu[:, 1] + q)))
    coverage = in_the_range / len(target) * 100
    
    return coverage
    # total_cov = 0.0
    # for mu_single, target_single in zip(mu, target):
    #     if before:
    #         if mu_single[0] <= target_single <= mu_single[1]:
    #             total_cov += 1.0
    #     else:
    #         if mu_single[0] - q <= target_single <= mu_single[1] + q:
    #             total_cov += 1.0
            
    # return total_cov / len(mu)

def scale_bins_single_conformal(mu_test, q):
    
    # Calculate Avg Length before temperature scaling
    before_scaling_avg_len = torch.mean(abs(mu_test[:, 1] - mu_test[:, 0]))
    # before_scaling_avg_len = (mu_test[1] - mu_test[0]).mean()
    print('Before scaling - Avg Length: %.3f' % (before_scaling_avg_len))
        
    # Calculate Avg Length after single scaling
    after_single_scaling_avg_len = torch.mean(abs((mu_test[:, 1] + q) - (mu_test[:, 0] - q)))
    # after_single_scaling_avg_len = ((mu_test[1] + q) - (mu_test[0] - q)).mean()
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
        calc_mean = False
        eval_test_set(data_dir, save_params=save_params, mix_indices=mix_indices, load_params=load_params, calc_mean=calc_mean, save_test=save_test, load_test=load_test)
    
    
def eval_test_set(data_dir="C:\lior\studies\master\projects\calibration/regression calibration/rsna-bone-age", save_params=False, load_params=False, mix_indices=True, calc_mean=False, save_test=False, load_test=False):
    base_model = 'densenet201'
    assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
    device = torch.device("cuda:0")
    
    if use_gauss_model:
        model = BreastPathQModelGauss(base_model, in_channels=1, out_channels=1,
                                pretrained=False).to(device)
    else:
    
        model = BreastPathQModel(base_model, in_channels=1, out_channels=2).to(device)

    # checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\{base_model}_gaussian_oct_best_freeze_lr_0.0003_nll.pth.tar")[0]
    # checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\cqr\{base_model}_0.95_boneage_cqr_best_new.pth.tar")[0]
    checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\cqr\{base_model}_0.95_boneage_lr_3e-05_cqr_best_new_gauss_pre.pth.tar")[0]
    # checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\{base_model}_gaussian_oct_315.pth.tar")[0]

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
        t_p_calib = []
        targets_calib = []
        
        if load_params:
            load_path = 'C:/lior/studies/master/projects/calibration/regression calibration/regression_calibration/reports/var_and_mse_calib/'
            with open(load_path + f'{base_model}_gaussian_boneage_calib_params_cqr_095.pickle', 'rb') as handle:
                calib_dict = pickle.load(handle)
                mu_calib = calib_dict['mu']
                target_calib = calib_dict['target']
                
        else:

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(tqdm(calib_loader)):
                    data, target = data.to(device), target.to(device)

                    if use_gauss_model:
                        t_low, t_high, _ = model(data, dropout=True, mc_dropout=True, test=True)
                        t_p = torch.cat((t_low, t_high), -1)
                    else:
                        t_p = model(data, dropout=True, mc_dropout=True, test=True)

                    t_p_calib.append(t_p.detach())
                    targets_calib.append(target.detach())
            
            
            t_p_calib = torch.cat(t_p_calib, dim=1).clamp(0, 1).permute(1,0,2)
            mu_calib = t_p_calib.mean(dim=1)
            target_calib = torch.cat(targets_calib, dim=0)
        
            if save_params:
                save_path = 'C:/lior/studies/master/projects/calibration/regression calibration/regression_calibration/reports/var_and_mse_calib/'
                with open(save_path + f'{base_model}_gaussian_boneage_calib_params_cqr_095_pre.pickle', 'wb') as handle:
                    pickle.dump({'mu': mu_calib,
                                'target': target_calib}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        if load_test:
            load_path = 'C:/lior/studies/master/projects/calibration/regression calibration/regression_calibration/reports/var_and_mse_calib/'
            with open(load_path + f'{base_model}_gaussian_boneage_test_results_cqr_095.pickle', 'rb') as handle:
                calib_dict = pickle.load(handle)
                t_p_test_list = calib_dict['tp']
                mu_test_list = calib_dict['mu']
                target_test_list = calib_dict['target']
                
        else:
        
            t_p_test_list = []
            mu_test_list = []
            target_test_list = []

            for i in range(1):
                t_p_test = []
                targets_test = []

                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
                        data, target = data.to(device), target.to(device)

                        if use_gauss_model:
                            t_low, t_high, _ = model(data, dropout=True, mc_dropout=True, test=True)
                            t_p = torch.cat((t_low, t_high), -1)
                        else:
                            t_p = model(data, dropout=True, mc_dropout=True, test=True)

                        t_p_test.append(t_p.detach())
                        targets_test.append(target.detach())

                    t_p_test = torch.cat(t_p_test, dim=1).clamp(0, 1).permute(1,0,2)
                    mu_test = t_p_test.mean(dim=1)
                    target_test = torch.cat(targets_test, dim=0)

                    t_p_test_list.append(t_p_test)
                    mu_test_list.append(mu_test)
                    target_test_list.append(target_test)
                    
            if save_test:
                save_path = 'C:/lior/studies/master/projects/calibration/regression calibration/regression_calibration/reports/var_and_mse_calib/'
                with open(save_path + f'{base_model}_gaussian_boneage_test_results_cqr_095_pre.pickle', 'wb') as handle:
                    pickle.dump({'tp': t_p_test_list, 'mu': mu_test_list, 'target': target_test_list}, handle, protocol=pickle.HIGHEST_PROTOCOL)

                
        # CQR
        avg_len_before_list = []
        avg_len_single_list = []

        avg_cov_before_list = []
        avg_cov_after_single_list = []
        
        alpha = 0.1
        
        for i in range(len(mu_test_list)):
            q = set_scaler_conformal(target_calib, mu_calib, alpha=alpha)
                        
            avg_len_single, avg_len_before = scale_bins_single_conformal(mu_test_list[i], q)
            
            avg_cov_before = avg_cov(mu_test_list[i], q, target_test_list[i].mean(dim=1), before=True)
            avg_cov_after_single = avg_cov(mu_test_list[i], q, target_test_list[i].mean(dim=1))
            
            avg_len_before_list.append(avg_len_before.cpu())
            avg_len_single_list.append(avg_len_single.cpu())
            
            avg_cov_before_list.append(avg_cov_before)
            avg_cov_after_single_list.append(avg_cov_after_single)
            
        if calc_mean:
            top_limit = mu_test_list[0][:, 1] + q
            bottom_limit = mu_test_list[0][:, 0] - q
            
            pred = (top_limit + bottom_limit) / 2
            mse = torch.nn.functional.mse_loss(pred, target_test_list[i].mean(dim=1))
            
        print(f'Test before, Avg Length:', torch.stack(avg_len_before_list).mean().item())
        print(f'Test after single, Avg Length:', torch.stack(avg_len_single_list).mean().item())

        print(f'Test before with Avg Cov:', torch.tensor(avg_cov_before_list).mean().item())
        print(f'Test after single with Avg Cov:', torch.tensor(avg_cov_after_single_list).mean().item())
        
        if calc_mean:
            print(f'Test MSE CQR:', mse.item())
        
        q_all.append(q.item())
        avg_len_all.append(torch.stack(avg_len_single_list).mean().item())
        avg_cov_all.append(torch.tensor(avg_cov_after_single_list).mean().item())
        
    print(q_all)
    print(avg_len_all)
    print(avg_cov_all)
    
    if len(q_all) > 1:
        print(f'q mean: {statistics.mean(q_all)}, q std: {statistics.stdev(q_all)}')
        print(f'avg_len mean: {statistics.mean(avg_len_all)}, avg_len std: {statistics.stdev(avg_len_all)}')
        print(f'avg_cov mean: {statistics.mean(avg_cov_all)}, avg_cov std: {statistics.stdev(avg_cov_all)}')
    
    
def eval_single(data_dir, image_path, labels_min=1.0, labels_max=228.0):
    base_model = 'densenet201'
    assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
    device = torch.device("cuda:0")
    
    alpha = 0.1
    
    model = BreastPathQModel(base_model, in_channels=1, out_channels=2).to(device)

    checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\cqr\{base_model}_0.95_boneage_cqr_best_new.pth.tar")[0]

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

        t_p = model(data, dropout=True, mc_dropout=True, test=True)
        
        t_p_test = t_p.detach().clamp(0, 1).permute(1,0,2)
        mu_test = t_p_test.mean(dim=1)
        
    load_path = 'C:/lior/studies/master/projects/calibration/regression calibration/regression_calibration/reports/var_and_mse_calib/'
    with open(load_path + f'{base_model}_gaussian_boneage_calib_params_cqr_095.pickle', 'rb') as handle:
        calib_dict = pickle.load(handle)
        mu_calib = calib_dict['mu']
        target_calib = calib_dict['target']
        
    q = calc_optimal_q(target_calib.mean(dim=1), mu_calib, alpha=alpha)
    
    top_limit = mu_test[:, 1] + q
    bottom_limit = mu_test[:, 0] - q
    
    print(f'Test target for {image_name}:', target.item())
    
    print(f'Test top limit CQR for {image_name}:', top_limit.item())
    print(f'Test top limit CQR for {image_name}:',bottom_limit.item())
    
if __name__ == '__main__':
    main()