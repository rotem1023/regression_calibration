import numpy as np
import os
import sys
# np.random.seed(1)
import torch
# torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import SubsetRandomSampler, ConcatDataset, Subset
from tqdm import tqdm
import torch
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from data_generator_endovis import EndoVisDataset
from data_generator_lumbar import LumbarDataset

from cqr_model import BreastPathQModel
from glob import glob
import statistics
import math
from skimage import io
import pandas as pd
import pickle


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
    print("Current PID:", os.getpid())
    eval_single_img = False
    data_dir = f"C:\lior\studies\master\projects\calibration/regression calibration/Tracking_Robotic_Testing/Tracking"
    
    if eval_single_img:
        image_name = '0285' # 7246 / 6956 / 11613
        image_path = f"C:\lior\studies\master\projects\calibration/regression calibration/Tracking_Robotic_Testing/Tracking/Dataset3/frames/{image_name}.png"
        eval_single(data_dir, image_path)
    else:
        mix_indices = False
        save_params = False
        load_params = False
        save_test = True
        load_test = False
        calc_mean = False
        eval_test_set(save_params=save_params, mix_indices=mix_indices, load_params=load_params, calc_mean=calc_mean, save_test=save_test, load_test=load_test)

def eval_test_set(level =1, save_params=False, load_params=False, mix_indices=True, calc_mean=False, save_test=False, load_test=False):
    # efficientnetb4, densenet201
    base_model = 'densenet201'
    level = 2
    models_dir = '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots/cqr'
    assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
    device = torch.device("cuda:3")
    iters = 20
    alpha = 0.1
    
    print(f'Running CQR for model {base_model} with alpha {alpha} and level {level}, {iters} iterations')
    
    model = BreastPathQModel(base_model, out_channels=2).to(device)

    # checkpoint_path = glob(f"/home/dsi/frenkel2/regression_calibration/models/{base_model}_gaussian_endovis_199_new.pth.tar")[0]
    # checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\{base_model}_gaussian_endovis_199_new.pth.tar")[0]
    if alpha == 0.1:
        checkpoint = torch.load(f'{models_dir}/{base_model}_lumbar_L{level}_cqr_best_new.pth.tar', map_location=device)
    else:
        checkpoint = torch.load(f'{models_dir}/{base_model}_lumbar_L{level}_alpha_0.05_cqr_best_new.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    batch_size = 64

    # data_dir_val = r'C:\lior\studies\master\projects\calibration\regression calibration\Tracking_Robotic_Testing\Tracking'
    # data_dir_test = r'C:\lior\studies\master\projects\calibration\regression calibration\Tracking_Robotic_Testing\Tracking'
    data_set_valid_original = LumbarDataset(level=level, mode='val', augment=False, scale=0.5)
    data_set_test_original = LumbarDataset(level=level, mode='test', augment=False, scale=0.5)

    assert len(data_set_valid_original) > 0
    assert len(data_set_test_original) > 0
    print(len(data_set_valid_original))
    print(len(data_set_test_original))
    
    if mix_indices:
        # Combine the datasets into one
        combined_dataset = ConcatDataset([data_set_valid_original, data_set_test_original])
    
    q_all = []
    avg_len_all = []
    avg_cov_all = []

    for _ in range(iters):
        if mix_indices:
            all_indices = torch.tensor((range(0, len(combined_dataset))))
            idx = torch.randperm(all_indices.nelement())
            all_indices = all_indices.view(-1)[idx].view(all_indices.size())

            # Create subsets using the split_indices
            data_set_valid = Subset(combined_dataset, all_indices[:len(combined_dataset)//2])
            data_set_test = Subset(combined_dataset, all_indices[len(combined_dataset)//2:])
            
        else:
            data_set_valid = data_set_valid_original
            data_set_test = data_set_test_original
        
        calib_loader = torch.utils.data.DataLoader(data_set_valid, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(data_set_test, batch_size=batch_size, shuffle=False)
        
        model.eval()
        t_p_calib = []
        targets_calib = []
        
        if load_params:
            load_path = 'C:/lior/studies/master/projects/calibration/regression calibration/regression_calibration/reports/var_and_mse_calib/'
            with open(load_path + f'{base_model}_gaussian_endovis_calib_params_cqr_095.pickle', 'rb') as handle:
                calib_dict = pickle.load(handle)
                mu_calib = calib_dict['mu']
                target_calib = calib_dict['target']

        else:
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(tqdm(calib_loader)):
                    data, target = data.to(device), target.to(device)

                    t_p = model(data, dropout=True, mc_dropout=True, test=True)

                    t_p_calib.append(t_p.detach())
                    targets_calib.append(target.detach())
            
            
            t_p_calib = torch.cat(t_p_calib, dim=1).clamp(0, 1).permute(1,0,2)
            mu_calib = t_p_calib.mean(dim=1)
            target_calib = torch.cat(targets_calib, dim=0)
            
        
        t_p_test_list = []
        mu_test_list = []
        target_test_list = []

        for i in range(1):
            t_p_test = []
            targets_test = []

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
                    data, target = data.to(device), target.to(device)

                    t_p = model(data, dropout=True, mc_dropout=True, test=True)

                    t_p_test.append(t_p.detach())
                    targets_test.append(target.detach())

                t_p_test = torch.cat(t_p_test, dim=1).clamp(0, 1).permute(1,0,2)
                mu_test = t_p_test.mean(dim=1)
                target_test = torch.cat(targets_test, dim=0)

                t_p_test_list.append(t_p_test)
                mu_test_list.append(mu_test)
                target_test_list.append(target_test)
                
        # CQR
        avg_len_before_list = []
        avg_len_single_list = []

        avg_cov_before_list = []
        avg_cov_after_single_list = []

        for i in range(len(mu_test_list)):
            q = set_scaler_conformal(target_calib, mu_calib, alpha = alpha)
                        
            avg_len_single, avg_len_before = scale_bins_single_conformal(mu_test_list[i], q)
            
            avg_cov_before = avg_cov(mu_test_list[i], q, target_test_list[i].mean(dim=1), before=True)
            avg_cov_after_single = avg_cov(mu_test_list[i], q, target_test_list[i].mean(dim=1))
            
            avg_len_before_list.append(avg_len_before.cpu())
            avg_len_single_list.append(avg_len_single.cpu())
            
            avg_cov_before_list.append(avg_cov_before)
            avg_cov_after_single_list.append(avg_cov_after_single)
            
        print(f'Test before, Avg Length:', get_float(torch.stack(avg_len_before_list).mean()))
        print(f'Test after single, Avg Length:', get_float(torch.stack(avg_len_single_list).mean()))

        print(f'Test before with Avg Cov:', get_float(torch.tensor(avg_cov_before_list).mean()))
        print(f'Test after single with Avg Cov:', get_float(torch.tensor(avg_cov_after_single_list).mean()))
        
        q_all.append(get_float(q))
        avg_len_all.append(get_float(torch.stack(avg_len_single_list).mean()))
        avg_cov_all.append(get_float(torch.tensor(avg_cov_after_single_list).mean()))
        
    print(q_all)
    print(avg_len_all)
    print(avg_cov_all)
    
    output_dir= '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results/cqr'
    output_file = f"lumbar_dataset_model_{base_model}_alpha_{alpha}_level_{level}_iterations_{iters}_cqr.txt"

    # Open the file in write mode
    with open(f'{output_dir}/{output_file}', 'w') as f:
        def log_and_print(message):
            print(message)          # Print to console
            f.write(message + '\n') # Write to file

        # q_all statistics
        if q_all:
            log_and_print(f'q mean: {statistics.mean(q_all):.4f}, q std: {statistics.stdev(q_all):.4f}' if len(q_all) > 1 
                        else f'q mean: {q_all[0]:.4f}, q std: undefined (only one value)')
        else:
            log_and_print('q_all is empty')

        # avg_len_all statistics
        if avg_len_all:
            log_and_print(f'avg_len mean: {statistics.mean(avg_len_all):.4f}, avg_len std: {statistics.stdev(avg_len_all):.4f}' if len(avg_len_all) > 1 
                        else f'avg_len mean: {avg_len_all[0]:.4f}, avg_len std: undefined (only one value)')
        else:
            log_and_print('avg_len_all is empty')

        # avg_cov_all statistics
        if avg_cov_all:
            log_and_print(f'avg_cov mean: {statistics.mean(avg_cov_all):.4f}, avg_cov std: {statistics.stdev(avg_cov_all):.4f}' if len(avg_cov_all) > 1 
                        else f'avg_cov mean: {avg_cov_all[0]:.4f}, avg_cov std: undefined (only one value)')
        else:
            log_and_print('avg_cov_all is empty')

    # print(f'q mean: {statistics.mean(q_all)}, q std: {statistics.stdev(q_all)}')
    # print(f'avg_len mean: {statistics.mean(avg_len_all)}, avg_len std: {statistics.stdev(avg_len_all)}')
    # print(f'avg_cov mean: {statistics.mean(avg_cov_all)}, avg_cov std: {statistics.stdev(avg_cov_all)}')


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

def eval_single(data_dir, image_path, labels_min=1.0, labels_max=228.0):
    base_model = 'densenet201'
    assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
    device = torch.device("cuda:0")
    
    alpha = 0.1
    
    model = BreastPathQModel(base_model, out_channels=2).to(device)

    checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\cqr\{base_model}_0.95_endovis_cqr_best_new.pth.tar")[0]

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("Loading previous weights at epoch " + str(checkpoint['epoch']) + " from\n" + checkpoint_path)
    model.eval()
    
    batch_size = 16
    
    x = io.imread(image_path)
    x = np.atleast_3d(x)
    x = to_pil_and_resize(x, 0.5)
    
    path = '/'.join(image_path.split('/')[:-2])
    df = pd.read_csv(path+'/Right_Instrument_Pose.txt', header=None, delim_whitespace=True)
    for index, row in df.iterrows():
        if row[0] > -1 and row[1] > -1:
            if f"{path}/frames/{(index+1):04}.png" == image_path:
                label = [row[0]/720, row[1]/576]
                break

    trans = transforms.ToTensor()

    x = trans(x).unsqueeze(0)
    
    label = torch.tensor(label).float()
    
    with torch.no_grad():
        data, target = x.to(device), label.to(device)

        t_p = model(data, dropout=True, mc_dropout=True, test=True)
        
        t_p_test = t_p.detach().clamp(0, 1).permute(1,0,2)
        mu_test = t_p_test.mean(dim=1)
        
    load_path = 'C:/lior/studies/master/projects/calibration/regression calibration/regression_calibration/reports/var_and_mse_calib/'
    with open(load_path + f'{base_model}_gaussian_endovis_calib_params_cqr_095.pickle', 'rb') as handle:
        calib_dict = pickle.load(handle)
        mu_calib = calib_dict['mu']
        target_calib = calib_dict['target']
        
    q = calc_optimal_q(target_calib.mean(dim=1), mu_calib, alpha=alpha)
    
    top_limit = mu_test[:, 1] + q
    bottom_limit = mu_test[:, 0] - q
    
    image_name = int(image_path.split('/')[-1].split('.')[0])
    
    print(f'Test target for {image_name}:', get_float(target[0]))
    
    print(f'Test top limit CQR for {image_name}:', get_float(top_limit))
    print(f'Test bottom limit CQR for {image_name}:',get_float(bottom_limit))
    
    
if __name__ == '__main__':
    main()