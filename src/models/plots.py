from matplotlib import pyplot as plt
import pickle

from calibration_plots import plot_uncert, plot_uncert_multi
from uce import uceloss
from scaling import set_scaler, scale_bins, calc_vars_mse_bins


def plot_uce_per_bin(base_model):
    save_path = '/dsi/scratch/from_netapp/users/frenkel2/data/calibration/well-calibrated-regression-uncertainty/results_uce_bins/'
    with open(save_path + f'{base_model}_gaussian_boneage_493.pickle', 'rb') as handle:
        uce_dict = pickle.load(handle)
        
    uce_per_bin_before = uce_dict['before'][-1]
    uce_per_bin_after_sigma = uce_dict['after'][-1]
    
    """
    plt.figure()
    plt.scatter(range(len(uce_per_bin_before)), uce_per_bin_before)
    plt.savefig(save_path + f'uce_per_bin_{base_model}_gaussian_boneage_493_before.png')
    plt.close()
    
    plt.figure()
    plt.scatter(range(len(uce_per_bin_after_sigma)), uce_per_bin_after_sigma)
    plt.savefig(save_path + f'uce_per_bin_{base_model}_gaussian_boneage_493_after_sigma.png')
    plt.close()
    """
    
    err_in_bin_before = uce_dict['before'][1].cpu()
    err_in_bin_after_sigma = uce_dict['after'][1].cpu()
    avg_uncert_in_bin_before = uce_dict['before'][2].cpu()
    avg_uncert_in_bin_after_sigma = uce_dict['after'][2].cpu()
    
    plt.figure()
    plt.scatter(range(len(err_in_bin_before)), err_in_bin_before, label='MSE')
    plt.scatter(range(len(avg_uncert_in_bin_before)), avg_uncert_in_bin_before, label='MV')
    plt.legend()
    plt.title('Uncalibrated')
    plt.savefig(save_path + f'uncert_error_per_bin_{base_model}_gaussian_boneage_499_before.png')
    plt.close()
    
    plt.figure()
    plt.scatter(range(len(err_in_bin_after_sigma)), err_in_bin_after_sigma, label='MSE')
    plt.scatter(range(len(avg_uncert_in_bin_after_sigma)), avg_uncert_in_bin_after_sigma, label='MV')
    plt.legend()
    plt.title('After sigma scaling')
    plt.savefig(save_path + f'uncert_error_per_bin_{base_model}_gaussian_boneage_499_after_sigma.png')
    plt.close()
    
def plot_s_per_bin(base_model, ds):
    save_path = '/dsi/scratch/from_netapp/users/frenkel2/data/calibration/well-calibrated-regression-uncertainty/results_uce_bins/'
    with open(save_path + f'{base_model}_gaussian_{ds}_s_bins.pickle', 'rb') as handle:
        s_dict = pickle.load(handle)
        
    bins_S = s_dict['s_bins'].cpu()
    single_S = s_dict['s'].cpu()
    
    plt.figure()
    plt.scatter(range(len(bins_S)), bins_S, label='Scaler per bin')
    plt.plot(range(len(bins_S)), [single_S]*len(bins_S), label='Single scaler', color='r')
    plt.legend()
    plt.title('Scaler per bin')
    plt.savefig(save_path + f'scaler_per_bin_{base_model}_gaussian_{ds}.png')
    plt.close()
    
def plot_calib_plots(base_model, ds, calib_mode, metric):
    save_path = '/dsi/scratch/from_netapp/users/frenkel2/data/calibration/well-calibrated-regression-uncertainty/results_uce_bins/'
    with open(save_path + f'{base_model}_gaussian_{ds}.pickle', 'rb') as handle:
        calib_dict = pickle.load(handle)
        
    err_calib = calib_dict['err'][0]
    err_test = calib_dict['err'][1]
    uncert_calib = calib_dict['uncert'][0]
    uncert_test = calib_dict['uncert'][1]
    uncert_calib_laves = calib_dict['uncert'][2]
    uncert_test_laves = calib_dict['uncert'][3]
    
    # if calib_mode == 'uncal':
    uce, err_in_bin, avg_sigma_in_bin, freq_in_bin, _ = uceloss(err_test[0]**2, uncert_test[0]**2, n_bins=6)
    plot_uncert(err_in_bin.cpu(), avg_sigma_in_bin.cpu(), save_path=save_path + f'mse_vs_mv_{base_model}_gaussian_{ds}_uncal.pdf', max_val=0.06, min_val=0.0)
    # elif calib_mode == 'vs':
    S = (err_calib**2 / uncert_calib**2).mean().sqrt()
    uce, err_in_bin, avg_sigma_in_bin, freq_in_bin, _ = uceloss(err_test[0]**2, (S*uncert_test[0])**2, n_bins=6)
    plot_uncert(err_in_bin.cpu(), avg_sigma_in_bin.cpu(), save_path=save_path + f'mse_vs_mv_{base_model}_gaussian_{ds}_vs.pdf', max_val=0.06, min_val=0.0)
    # elif calib_mode == 'bvs':
    n_bins = 6
    bins_T, S, bin_boundaries, uce = set_scaler(err_calib, uncert_calib, num_bins=n_bins, cross_validate=metric)
    # uce, uncert_test_after = scale_bins(err_test[0], uncert_test[0], bins_T, bin_boundaries, num_bins=n_bins)
    # uce, err_in_bin, avg_sigma_in_bin, freq_in_bin, _ = uceloss(err_test[0]**2, uncert_test_after**2)
    uce, uncert_test_after = scale_bins(err_calib, uncert_calib, bins_T, bin_boundaries, num_bins=n_bins)
    uce, err_in_bin, avg_sigma_in_bin, freq_in_bin, _ = uceloss(err_calib**2, uncert_test_after**2, n_bins=n_bins)
    plot_uncert(err_in_bin.cpu(), avg_sigma_in_bin.cpu(), save_path=save_path + f'mse_vs_mv_{base_model}_gaussian_{ds}_bvs_{metric}_val.pdf', max_val=0.06, min_val=0.0)
    # plot_uncert(err_in_bin.cpu(), avg_sigma_in_bin.cpu(), save_path=save_path + f'mse_vs_mv_{base_model}_gaussian_{ds}_{calib_mode}.pdf', max_val=0.08387443, min_val=0.0)
    print(uce.item()*100)
    
def plot_var_dist_bins(base_model, ds, mode='test'):
    save_path = '/dsi/scratch/from_netapp/users/frenkel2/data/calibration/well-calibrated-regression-uncertainty/results_uce_bins/'
    with open(save_path + f'{base_model}_gaussian_{ds}.pickle', 'rb') as handle:
        calib_dict = pickle.load(handle)
        
    err_calib = calib_dict['err'][0]
    err_test = calib_dict['err'][1]
    uncert_calib = calib_dict['uncert'][0]
    uncert_test = calib_dict['uncert'][1]
    uncert_calib_laves = calib_dict['uncert'][2]
    uncert_test_laves = calib_dict['uncert'][3]
    
    if mode == 'test':
        errors_in_bin_list, uncert_in_bin_list = calc_vars_mse_bins(err_test[0]**2, uncert_test[0]**2)
    elif mode == 'val':
        errors_in_bin_list, uncert_in_bin_list = calc_vars_mse_bins(err_calib**2, uncert_calib**2)
    
    i = 0
    for errors, uncert in zip(errors_in_bin_list, uncert_in_bin_list):
        plt.figure()
        plt.scatter(range(len(errors.cpu())), errors.cpu())
        plt.title(f'Vars in bin {i}')
        plt.savefig(save_path + f'vars_{base_model}_gaussian_{ds}_{mode}_bin_{i}.png')
        plt.close()
        
        plt.figure()
        plt.scatter(range(len(uncert.cpu())), uncert.cpu())
        plt.title(f'MSEs in bin {i}')
        plt.savefig(save_path + f'mses_{base_model}_gaussian_{ds}_{mode}_bin_{i}.png')
        plt.close()
        
        i += 1

if __name__ == '__main__':
    metric = 'uce'
    calib_mode = 'bvs'
    ds = 'boneage'
    base_model = 'efficientnetb4'
    # plot_uce_per_bin(base_model)
    # plot_s_per_bin(base_model, ds)
    plot_calib_plots(base_model, ds, calib_mode, metric)