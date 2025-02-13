import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_hist(dataset, model, dist_model, scale_factor, lambada_param, d_plus, d_minus, true_distance):
    plt.figure(figsize=(12, 5))
    # Histogram for d+
    plt.subplot(1, 2, 1)
    plt.hist(d_plus, bins=30, alpha=0.7, color='b', label="d+")
    plt.xlabel("Distance Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of d+")
    plt.legend()

    # Histogram for d-
    plt.subplot(1, 2, 2)
    plt.hist(d_minus, bins=30, alpha=0.7, color='r', label="d-")
    plt.xlabel("Distance Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of d-")
    plt.legend()

    plt.tight_layout()
    st_true = "true" if true_distance else "prediction"
    plt.savefig(f"src/models/results_new/predictions_mse/{dataset}/mse/{model}/{dist_model}/valid/{scale_factor}/{lambada_param}histogram_{st_true}.png", dpi=300)  # Save the figure
    plt.show()

# 'resnet101', 'densenet201', 'efficientnetb4'

dataset = 'boneage'
model = 'densenet201'
dist_model = 'resnet50'
loss = 'mse'
cur_dir= 'predictions_mse'
scale_factor = 1
lambada_param = 1

y = np.load(f"src/models/results_new/{cur_dir}/{dataset}/{loss}/{model}/{dist_model}/valid/{scale_factor}/{lambada_param}/y.npy").squeeze()
mu = np.load(f"src/models/results_new/{cur_dir}/{dataset}/{loss}/{model}/{dist_model}/valid/{scale_factor}/{lambada_param}/mu.npy").squeeze()


d_minus_true = np.where(mu > y, mu - y, 0)
d_plus_true = np.where(y > mu, y- mu, 0)

positive_d_pred = np.load(f"src/models/results_new/{cur_dir}/{dataset}/{loss}/{model}/{dist_model}/valid/{scale_factor}/{lambada_param}/positive_distance.npy").squeeze()
negative_distance_pred = np.load(f"src/models/results_new/{cur_dir}/{dataset}/{loss}/{model}/{dist_model}/valid/{scale_factor}/{lambada_param}/negative_distance.npy").squeeze()


print(f"dataset: {dataset} 90% d plus: {torch.quantile(torch.from_numpy(d_plus_true),0.9)}, 90% d minus: {torch.quantile(torch.from_numpy(d_minus_true), 0.9)}")
plot_hist(dataset=dataset, model=model, dist_model=dist_model, scale_factor=scale_factor, lambada_param=lambada_param, d_plus=positive_d_pred, d_minus=negative_distance_pred, true_distance=False)
plot_hist(dataset=dataset, model=model, dist_model=dist_model, scale_factor=scale_factor, lambada_param=lambada_param, d_plus=d_plus_true, d_minus=d_minus_true, true_distance=True)
