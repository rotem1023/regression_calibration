import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# def plot_hist(dataset, model, dist_model, scale_factor, lambada_param, d_plus, d_minus, true_distance):
#     plt.figure(figsize=(12, 5))
#     # Histogram for d+
#     plt.subplot(1, 2, 1)
#     plt.hist(d_plus, bins=30, alpha=0.7, color='b', label="d+")
#     plt.xlabel("Distance Value")
#     plt.ylabel("Frequency")
#     plt.title("Histogram of d+")
#     plt.legend()

#     # Histogram for d-
#     plt.subplot(1, 2, 2)
#     plt.hist(d_minus, bins=30, alpha=0.7, color='r', label="d-")
#     plt.xlabel("Distance Value")
#     plt.ylabel("Frequency")
#     plt.title("Histogram of d-")
#     plt.legend()

#     plt.tight_layout()
#     st_true = "true" if true_distance else "prediction"
#     plt.savefig(f"src/models/results_new/predictions_mse/{dataset}/mse/{model}/{dist_model}/valid/{scale_factor}/{lambada_param}histogram_{st_true}.png", dpi=300)  # Save the figure
#     plt.show()

# # 'resnet101', 'densenet201', 'efficientnetb4'

# dataset = 'boneage'
# model = 'densenet201'
# dist_model = 'resnet50'
# loss = 'mse'
# cur_dir= 'predictions_mse'
# scale_factor = 1
# lambada_param = 1

# y = np.load(f"src/models/results_new/{cur_dir}/{dataset}/{loss}/{model}/{dist_model}/valid/{scale_factor}/{lambada_param}/y.npy").squeeze()
# mu = np.load(f"src/models/results_new/{cur_dir}/{dataset}/{loss}/{model}/{dist_model}/valid/{scale_factor}/{lambada_param}/mu.npy").squeeze()


# d_minus_true = np.where(mu > y, mu - y, 0)
# d_plus_true = np.where(y > mu, y- mu, 0)

# positive_d_pred = np.load(f"src/models/results_new/{cur_dir}/{dataset}/{loss}/{model}/{dist_model}/valid/{scale_factor}/{lambada_param}/positive_distance.npy").squeeze()
# negative_distance_pred = np.load(f"src/models/results_new/{cur_dir}/{dataset}/{loss}/{model}/{dist_model}/valid/{scale_factor}/{lambada_param}/negative_distance.npy").squeeze()


# print(f"dataset: {dataset} 90% d plus: {torch.quantile(torch.from_numpy(d_plus_true),0.9)}, 90% d minus: {torch.quantile(torch.from_numpy(d_minus_true), 0.9)}")
# plot_hist(dataset=dataset, model=model, dist_model=dist_model, scale_factor=scale_factor, lambada_param=lambada_param, d_plus=positive_d_pred, d_minus=negative_distance_pred, true_distance=False)
# plot_hist(dataset=dataset, model=model, dist_model=dist_model, scale_factor=scale_factor, lambada_param=lambada_param, d_plus=d_plus_true, d_minus=d_minus_true, true_distance=True)

def get_saved_dir(results_dir, dataset, base_model, dist_model, loss,group,  level = None, lambda_param = 1, scale_factor = 1):
    cur_dir = results_dir
    os.makedirs(cur_dir, exist_ok=True)
    cur_dir = f"{cur_dir}/{dataset}"
    os.makedirs(cur_dir, exist_ok=True)
    if level != None:
        cur_dir = f"{cur_dir}/{level}"
        os.makedirs(cur_dir, exist_ok=True)
    cur_dir = f"{cur_dir}/{loss}"
    os.makedirs(cur_dir, exist_ok=True)
    cur_dir = f"{cur_dir}/{base_model}"
    os.makedirs(cur_dir, exist_ok=True)
    cur_dir = f"{cur_dir}/{dist_model}"
    os.makedirs(cur_dir, exist_ok=True)
    cur_dir = f"{cur_dir}/{group}"
    os.makedirs(cur_dir, exist_ok=True)
    cur_dir = f"{cur_dir}/{lambda_param}"
    os.makedirs(cur_dir, exist_ok=True)
    cur_dir =f"{cur_dir}/{int(scale_factor)}"
    os.makedirs(cur_dir, exist_ok=True)
    return cur_dir

def load_arrays(results_dir, dataset, base_model, dist_model, loss, group, level, lambda_param, scale_factor):
    saved_dir = get_saved_dir(results_dir=results_dir, dataset=dataset, base_model=base_model, dist_model=dist_model, loss = loss, group = group, level=level, lambda_param = lambda_param, scale_factor = scale_factor)
    y = np.load(f'{saved_dir}/y.npy')
    mu = np.load(f'{saved_dir}/mu.npy')
    logvar = np.load(f'{saved_dir}/logvar.npy')  
    pos_dist = np.load(f'{saved_dir}/positive_distance.npy') 
    neg_dist = np.load(f'{saved_dir}/negative_distance.npy')
    return torch.from_numpy(y), torch.from_numpy(mu), torch.from_numpy(logvar), torch.from_numpy(pos_dist), torch.from_numpy(neg_dist)

import numpy as np
import matplotlib.pyplot as plt

def plot_sorted_bounds(targets, my_upper, my_lower, other_upper, other_lower, filename="plot.png"):
    """
    Sorts input arrays based on targets and plots the upper and lower bounds for two methods.
    Saves the plot to a file.

    Parameters:
        targets (np.array): The target values (used for sorting).
        my_upper (np.array): Upper bound for "My Method".
        my_lower (np.array): Lower bound for "My Method".
        other_upper (np.array): Upper bound for "Other Method".
        other_lower (np.array): Lower bound for "Other Method".
        filename (str): File path to save the plot (default: "plot.png").
    """
    # Sort indices based on targets
    sorted_indices = np.argsort(targets)
    end = len(sorted_indices)
    start = 0
    end = 100
    start = 50

    # Sort all arrays
    targets_sorted = targets[sorted_indices][start:end]
    my_upper_sorted = my_upper[sorted_indices][start:end]
    my_lower_sorted = my_lower[sorted_indices][start:end]
    other_upper_sorted = other_upper[sorted_indices][start:end]
    other_lower_sorted = other_lower[sorted_indices][start:end]
    sorted_indices = np.arange(start, end)
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_indices, my_upper_sorted, label="My Method Upper", linestyle="--", marker="o")
    plt.plot(sorted_indices, my_lower_sorted, label="My Method Lower", linestyle="--", marker="o")
    plt.plot(sorted_indices, other_upper_sorted, label="Other Method Upper", linestyle=":", marker="s")
    plt.plot(sorted_indices, other_lower_sorted, label="Other Method Lower", linestyle=":", marker="s")
    plt.plot(sorted_indices, targets_sorted, label="Targets", linestyle="-.", color="black", alpha=0.5)

    # Labels and legend
    plt.xlabel("Targets (Sorted)")
    plt.ylabel("Values")
    plt.title("Comparison of Methods")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save plot to file
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to free memory
    
def plot_sorted_preds(targets, my_upper, my_lower, other_upper, other_lower, filename="plot.png"):
    """
    Sorts input arrays based on targets and plots the upper and lower bounds for two methods.
    Saves the plot to a file.

    Parameters:
        targets (np.array): The target values (used for sorting).
        my_upper (np.array): Upper bound for "My Method".
        my_lower (np.array): Lower bound for "My Method".
        other_upper (np.array): Upper bound for "Other Method".
        other_lower (np.array): Lower bound for "Other Method".
        filename (str): File path to save the plot (default: "plot.png").
    """
    # Sort indices based on targets
    sorted_indices = np.argsort(targets)
    end = len(sorted_indices)
    start = 0
    end = 1350
    start = 1300

    # Sort all arrays
    targets_sorted = targets[sorted_indices][start:end]
    my_upper_sorted = my_upper[sorted_indices][start:end]
    my_lower_sorted = my_lower[sorted_indices][start:end]
    other_upper_sorted = other_upper[sorted_indices][start:end]
    other_lower_sorted = other_lower[sorted_indices][start:end]
    sorted_indices = np.arange(start, end)
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_indices, my_upper_sorted, label="My Method Upper", linestyle="--", marker="o")
    plt.plot(sorted_indices, my_lower_sorted, label="My Method Lower", linestyle="--", marker="o")
    plt.plot(sorted_indices, other_upper_sorted, label="True Upper", linestyle=":", marker="s")
    plt.plot(sorted_indices, other_lower_sorted, label="True Lower", linestyle=":", marker="s")

    # Labels and legend
    plt.xlabel("Targets (Sorted)")
    plt.ylabel("Values")
    plt.title("Comparison of Methods")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save plot to file
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to free memory
    
base_model = 'efficientnetb4'
base_model_dist = 'resnet50'
assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
device = torch.device("cuda:2")
dataset = 'lumbar'
loss = 'gaussian'
pred_x = False
pred_y = False
one_output = False
normalize = False
scale_factor = 1.0
lambda_param = 1
iters = 20
level = 1
results_dir = '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results_new/predictions'


targets_calib_original, y_p_calib_original,  logvars_calib_original, positive_dist_calib_original, negative_dist_calib_original = load_arrays(results_dir = results_dir, dataset = dataset, base_model = base_model, dist_model = base_model_dist, loss = loss, group = 'valid', level = level, lambda_param=lambda_param, scale_factor = scale_factor)
vars_calib_original = logvars_calib_original.exp()
sd_calib_original = vars_calib_original.sqrt()
targets_calib_original = targets_calib_original.unsqueeze(1)

negative_dist_calib_original, positive_dist_calib_original = torch.min(negative_dist_calib_original, positive_dist_calib_original), torch.max(negative_dist_calib_original, positive_dist_calib_original)
    

my_method_upper_bound = positive_dist_calib_original
my_method_lower_bound = negative_dist_calib_original

other_method_upper_bound = y_p_calib_original + sd_calib_original
other_method_lower_bound = y_p_calib_original - sd_calib_original

true_d_plus = np.where(targets_calib_original > y_p_calib_original, targets_calib_original , y_p_calib_original)
true_d_minus = np.where(targets_calib_original > y_p_calib_original, y_p_calib_original, targets_calib_original)
 
plot_sorted_preds(targets_calib_original.squeeze(-1), my_method_upper_bound, my_method_lower_bound, true_d_plus.squeeze(-1), true_d_minus.squeeze(-1), "preds.png")
plot_sorted_preds(targets_calib_original.squeeze(-1), other_method_upper_bound, other_method_lower_bound, true_d_plus.squeeze(-1), true_d_minus.squeeze(-1), "preds_other.png")

 
plot_sorted_bounds(targets_calib_original.squeeze(-1), my_method_upper_bound, my_method_lower_bound, other_method_upper_bound, other_method_lower_bound)


    
    
    
    
    
    
    
    
    
    
    
