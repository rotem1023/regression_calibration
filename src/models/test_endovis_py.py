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
from models import BreastPathQModel
from glob import glob
import statistics
import math
import pickle
import pandas as pd
from skimage import io


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
    eval_single_img = True
    data_dir = f"C:\lior\studies\master\projects\calibration/regression calibration/Tracking_Robotic_Testing/Tracking"
    
    if eval_single_img:
        image_name = '0285' # 7246 / 6956 / 11613
        image_path = f"C:\lior\studies\master\projects\calibration/regression calibration/Tracking_Robotic_Testing/Tracking/Dataset3/frames/{image_name}.png"
        eval_single(data_dir, image_path)
    else:
        mix_indices = False
        save_params = True
        load_params = False
        save_test = True
        load_test = False
        calc_mean = False
        eval_test_set(data_dir, save_params=save_params, mix_indices=mix_indices, load_params=load_params, calc_mean=calc_mean, save_test=save_test, load_test=load_test)

def eval_test_set(data_dir=f"C:\lior\studies\master\projects\calibration/regression calibration/Tracking_Robotic_Testing/Tracking", save_params=False, load_params=False, mix_indices=True, calc_mean=False, save_test=False, load_test=False):
    base_model = 'densenet201'
    assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
    device = torch.device("cuda:0")
    
    alpha = 0.1
    
    model = BreastPathQModel(base_model, out_channels=2).to(device)

    # checkpoint_path = glob(f"/home/dsi/frenkel2/regression_calibration/models/{base_model}_gaussian_endovis_199_new.pth.tar")[0]
    checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\{base_model}_gaussian_endovis_199_new.pth.tar")[0]
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("Loading previous weights at epoch " + str(checkpoint['epoch']) + " from\n" + checkpoint_path)
    
    batch_size = 16

    # data_dir_val = '/home/dsi/frenkel2/data/Tracking_Robotic_Testing/Tracking'
    # data_dir_test = '/home/dsi/frenkel2/data/Tracking_Robotic_Testing/Tracking'
    data_set_valid_original = EndoVisDataset(data_dir=data_dir, mode='val', augment=False, scale=0.5)
    data_set_test_original = EndoVisDataset(data_dir=data_dir, mode='test', augment=False, scale=0.5)

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
    q_all_gc = []
    avg_len_all_gc = []
    avg_cov_all_gc = []

    for _ in range(1):
        if mix_indices:
            # Define the indices to split the dataset
            split_indices = [0, len(data_set_valid_original)]  # Split between dataset1 and dataset2

            # Create subsets using the split_indices
            data_set_valid = Subset(combined_dataset, range(split_indices[0], split_indices[1]))
            data_set_test = Subset(combined_dataset, range(split_indices[1], len(combined_dataset)))
            
        else:
            data_set_valid = data_set_valid_original
            data_set_test = data_set_test_original
        
        calib_loader = torch.utils.data.DataLoader(data_set_valid, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(data_set_test, batch_size=batch_size, shuffle=False)
        
        model.eval()
        y_p_calib = []
        vars_calib = []
        logvars_calib = []
        targets_calib = []
        
        if load_params:
            load_path = 'C:/lior/studies/master/projects/calibration/regression calibration/regression_calibration/reports/var_and_mse_calib/'
            with open(load_path + f'{base_model}_gaussian_endovis_calib_params.pickle', 'rb') as handle:
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
                with open(save_path + f'{base_model}_gaussian_endovis_calib_params.pickle', 'wb') as handle:
                    pickle.dump({'mu': mu_calib,
                                'target': target_calib,
                                'err': err_calib, 
                                'uncert': uncert_calib,
                                }
                                , handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        y_p_test_list = []
        mu_test_list = []
        var_test_list = []
        logvars_test_list = []
        logvar_test_list = []
        target_test_list = []

        for i in range(5):
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
                
        err_test = [(target_test-mu_test).pow(2).mean(dim=1, keepdim=True).sqrt() for target_test, mu_test in zip(target_test_list, mu_test_list)]
        errvar_test = [(y_p_test-target_test.unsqueeze(1).repeat(1,25,1)).pow(2).mean(dim=(1,2)).unsqueeze(-1) for target_test, y_p_test in zip(target_test_list, y_p_test_list)]

        uncert_aleatoric_test = [logvar_test.exp().mean(dim=1, keepdim=True) for logvar_test in logvar_test_list]
        uncert_epistemic_test = [var_test.mean(dim=1, keepdim=True) for var_test in var_test_list]

        if uncertainty == 'aleatoric':
            uncert_test = [uncert_aleatoric_t.sqrt().clamp(0, 1) for uncert_aleatoric_t in uncert_aleatoric_test]
            uncert_test_laves = [(u_a_t + u_e_t).sqrt().clamp(0, 1) for u_a_t, u_e_t in zip(uncert_aleatoric_test, uncert_epistemic_test)]
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
            
        print(f'Test before, Avg Length:', torch.stack(avg_len_before_list).mean().item())
        print(f'Test after single CP, Avg Length:', torch.stack(avg_len_single_list).mean().item())
        print(f'Test after single GC, Avg Length:', torch.stack(avg_len_single_list_gc).mean().item())

        print(f'Test before with Avg Cov:', torch.tensor(avg_cov_before_list).mean().item())
        print(f'Test after single CP with Avg Cov:', torch.tensor(avg_cov_after_single_list).mean().item())
        print(f'Test after single GC with Avg Cov:', torch.tensor(avg_cov_after_single_list_gc).mean().item())
        
        q_all.append(q.item())
        avg_len_all.append(torch.stack(avg_len_single_list).mean().item())
        avg_cov_all.append(torch.tensor(avg_cov_after_single_list).mean().item())
        
        q_all_gc.append(q_gc.item())
        avg_len_all_gc.append(torch.stack(avg_len_single_list_gc).mean().item())
        avg_cov_all_gc.append(torch.tensor(avg_cov_after_single_list_gc).mean().item())
        
    print(q_all)
    print(avg_len_all)
    print(avg_cov_all)
    print(q_all_gc)
    print(avg_len_all_gc)
    print(avg_cov_all_gc)

    print(f'q CP mean: {statistics.mean(q_all)}, q CP std: {statistics.stdev(q_all)}')
    print(f'avg_len CP mean: {statistics.mean(avg_len_all)}, avg_len CP std: {statistics.stdev(avg_len_all)}')
    print(f'avg_cov CP mean: {statistics.mean(avg_cov_all)}, avg_cov CP std: {statistics.stdev(avg_cov_all)}')
    
    print(f'q GC mean: {statistics.mean(q_all_gc)}, q GC std: {statistics.stdev(q_all_gc)}')
    print(f'avg_len GC mean: {statistics.mean(avg_len_all_gc)}, avg_len GC std: {statistics.stdev(avg_len_all_gc)}')
    print(f'avg_cov GC mean: {statistics.mean(avg_cov_all_gc)}, avg_cov GC std: {statistics.stdev(avg_cov_all_gc)}')
    
    print(f"endovis, {base_model}, {alpha}")
    
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

    checkpoint_path = glob(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration\models\snapshots\{base_model}_gaussian_endovis_199_new.pth.tar")[0]

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

        y_p, logvar, var_bayesian = model(data, dropout=True, mc_dropout=True, test=True)
        
        y_p_test = y_p.detach().clamp(0, 1).permute(1,0,2)
        mu_test = y_p_test.mean(dim=1)
        logvar_test = logvar.permute(1,0,2).mean(dim=1)
        uncert_test = logvar_test.exp().mean(dim=1, keepdim=True).sqrt().clamp(0, 1)
        
    load_path = 'C:/lior/studies/master/projects/calibration/regression calibration/regression_calibration/reports/var_and_mse_calib/'
    with open(load_path + f'{base_model}_gaussian_endovis_calib_params.pickle', 'rb') as handle:
        calib_dict = pickle.load(handle)
        uncert_calib = calib_dict['uncert']
        mu_calib = calib_dict['mu']
        target_calib = calib_dict['target']
        err_calib = calib_dict['err']
        
    target_calib = target_calib.mean(dim=1, keepdim=True)
    mu_calib = mu_calib.mean(dim=1, keepdim=True)

    q = set_scaler_conformal(target_calib, mu_calib, uncert_calib, err_calib=err_calib, gc=False, alpha=alpha)
    
    top_limit = mu_test + q * uncert_test
    bottom_limit = mu_test - q * uncert_test
    
    q_gc = set_scaler_conformal(target_calib, mu_calib, uncert_calib, err_calib=err_calib, gc=True, alpha=alpha)
    
    top_limit_gc = mu_test + q_gc * uncert_test
    bottom_limit_gc = mu_test - q_gc * uncert_test
    
    image_name = int(image_path.split('/')[-1].split('.')[0])
    
    plot_images(image_path, target, bottom_limit, top_limit, bottom_limit_gc, top_limit_gc, bottom_limit_cqr=0.73, top_limit_cqr=0.90)
    
    print(f'Test target for {image_name}: {target[0].item()}\n')
    
    print(f'Test top limit CP for {image_name}:', top_limit[0, 0].item())
    print(f'Test bottom limit CP for {image_name}:',bottom_limit[0, 0].item())

    print(f'Test top limit GC for {image_name}:', top_limit_gc[0, 0].item())
    print(f'Test bottom limit GC for {image_name}:',bottom_limit_gc[0, 0].item())
    
def plot_images(image_path, target, bottom_limit, top_limit, bottom_limit_gc, top_limit_gc, bottom_limit_cqr=0.47, top_limit_cqr=0.54):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    image_name = int(image_path.split('/')[-1].split('.')[0])
    img = mpimg.imread(image_path)
    plt.figure()
    plt.imshow(img)
    plt.scatter(target[0].item()*720, target[1].item()*576, marker='x', color='red', s=100)
    plt.errorbar(target[0].item()*720, target[1].item()*576, xerr=np.expand_dims(np.array([target[0].item()*720 - bottom_limit[0, 0].item()*720, top_limit[0, 0].item()*720 - target[0].item()*720]), -1), ecolor='b')
    plt.errorbar(target[0].item()*720, target[1].item()*576 - 30, xerr=np.expand_dims(np.array([target[0].item()*720 - bottom_limit_gc[0, 0].item()*720, top_limit_gc[0, 0].item()*720 - target[0].item()*720 + 30]), -1), ecolor='g')
    plt.errorbar(target[0].item()*720, target[1].item()*576 + 30, xerr=np.expand_dims(np.array([target[0].item()*720 - bottom_limit_cqr*720, top_limit_cqr*720 - target[0].item()*720 + 30]), -1), ecolor='k')
    legend_labels = ['Gaussian-VS', 'CP-VS', 'CQR']
    legend_colors = ['green', 'blue', 'black']
    plt.legend(handles=[
        plt.Line2D([0], [0], color='green', lw=2, label=legend_labels[0]),
        plt.Line2D([0], [0], color='blue', lw=2, label=legend_labels[1]),
        plt.Line2D([0], [0], color='black', lw=2, label=legend_labels[2]),
    ], loc='upper right')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"C:\lior\studies\master\projects\calibration/regression calibration/regression_calibration/reports/var_and_mse_calib/{image_name}_annote.pdf", dpi=40)
    plt.close()
    
    
if __name__ == '__main__':
    main()