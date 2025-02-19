import torch
import sys
from tqdm import tqdm
import pickle
sys.path.append("/home/dsi/rotemnizhar/dev/regression_calibration/src/models")
from models import DistancePredictor, BreastPathQModel
from data_generator_boneage import BoneAgeDataset
from data_generator_lumbar import LumbarDataset
from data_generator_oct import OCTDataset
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 'resnet101', 'densenet201', 'efficientnetb4'
dataset_name = 'lumbar_L5'
model_name = 'densenet201'
dist_model_name = 'resnet50'
loss = 'gaussian'
gpu = '3'
in_channels = 3
out_channels = 1
results_dir = '/home/dsi/rotemnizhar/dev/regression_calibration/notebooks/arrays'



device = f"cuda:{gpu}"



def get_dataset(dataset_name):
    if dataset_name=='boneage':
        data_set = BoneAgeDataset(group='valid')
    elif dataset_name=='oct':
        data_set = OCTDataset(group='valid') 
    else:
        data_set = LumbarDataset(level=1, mode='valid', augment=False, scale=0.5)
    return data_set


def get_arrays(data_loader, model, dist_model, device, dataset_name):
    targets_s = []
    y_p_s = []
    logvar_s = []
    distance_plus_s = []
    distance_minus_s = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            if data.shape[0] != 32:
                print("data.shape[0] != 32")
                continue

            if 'lum' == dataset_name[:3]:
                y_p, logvar, var_bayesian = model(data, dropout=True, mc_dropout=False, test=False)
            else:
                y_p, logvar, var_bayesian = model(data, dropout=True, mc_dropout=True, test=True)
            
            if dataset_name =='boneage':
                target = target.squeeze(-1)
            
            
            y_p_s.append(y_p.detach())
            targets_s.append(target.detach())
            logvar_s.append(logvar.cpu())
            
            distances = dist_model(data).detach()
            distance_plus_s.append(distances[:,0])
            distance_minus_s.append(distances[:,1])

    
    if dataset_name[:3] != 'lum':    
        y_p_s = torch.cat(y_p_s, dim=1).clamp(0, 1).permute(1,0,2)
        y_p_s = y_p_s.mean(dim=1)  
        logvar_s = torch.cat(logvar_s, dim=1).permute(1,0,2)
        logvar_s = logvar_s.mean(dim=1)
    else:
        y_p_s = torch.cat(y_p_s)
        logvar_s = torch.cat(logvar_s).cpu()
            
    return torch.cat(targets_s).cpu(), y_p_s.cpu(), logvar_s.cpu(), torch.cat(distance_plus_s).cpu(), torch.cat(distance_minus_s).cpu()

model = BreastPathQModel(model_name, in_channels=in_channels, out_channels=out_channels).to(device)
checkpoint_path = f"/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots/{model_name}_{loss}_{dataset_name}_best.pth.tar"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
print("Loading previous weights at epoch " + str(checkpoint['epoch']) + " from\n" + checkpoint_path)
    
dist_model_name = 'resnet50'
dist_model = DistancePredictor(dist_model_name, in_channels=in_channels).to(device)
checkpoint = torch.load(f'/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots_new/{dist_model_name}_{dataset_name}_snapshot_dist_{model_name}_lambda_1_new.pth.tar', map_location=device)
dist_model.load_state_dict(checkpoint['state_dict'])
dist_model.eval()

dataset = get_dataset(dataset_name)
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=False)

target, mu, logvar, predicted_distance_plus, predicted_distance_minus = get_arrays(loader, model, dist_model, device, dataset_name)
true_d_plus = torch.clamp(target - mu, min=0) # True d+
true_d_minus = torch.clamp(mu - target, min=0)  # True d-


data_to_save = {
    "target": target,
    "mu": mu,
    "logvar": logvar,
    "predicted_distance_plus": predicted_distance_plus,
    "predicted_distance_minus": predicted_distance_minus,
    "true_d_plus": true_d_plus,
    "true_d_minus": true_d_minus,
}

filename= f"{results_dir}/dataset_{dataset_name}_model_{model_name}_distnce_model_{dist_model_name}_loss_{loss}.pkl"
with open(filename, "wb") as f:
    pickle.dump(data_to_save, f)

print(f"Data saved to {filename}")