# Max-Heinrich Laves
# Institute of Mechatronic Systems
# Leibniz Universität Hannover, Germany
# 2019


from datetime import datetime
import os
import fire
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from tqdm import tqdm
# from data_generator_breast import BreastPathQDataset
from data_generator_boneage import BoneAgeDataset
from data_generator_lumbar import LumbarDataset
from data_generator_oct import OCTDataset
from models import BreastPathQModel, DistancePredictor, DistNewModel
from utils import kaiming_normal_init
from utils import nll_criterion_gaussian, nll_criterion_laplacian
import torch.nn as nn
import load_trained_models 
from torch.utils.data import Dataset, DataLoader


def save_snapshot(save_dir, model_name, dataset_name, epoch, model, dist_base_model_name,lambda_param, scale_factor,  is_best = False, is_mornalize =False):
    suffix = 'normalize_' if is_mornalize else ''
    suffix = suffix + ('best' if is_best else 'new')
    os.makedirs(save_dir, exist_ok=True)
    dist_str = f'dist_{dist_base_model_name}' if dist_base_model_name is not None else ''
    filename = f"{save_dir}/{model_name}_{dataset_name}_snapshot_{dist_str}_lambda_{int(lambda_param)}_dist_scale_factor{int(scale_factor)}_{suffix}.pth.tar"
    print(f"Saving snapshot, path: {filename}")
    torch.save({
        'epoch': epoch,
        'lambda_param': lambda_param,
        'state_dict': model.state_dict(),
    }, filename)    

torch.backends.cudnn.benchmark = True



def print_grad_norms(dist_model):
    # Get all the named parameters of the model
    named_params = list(dist_model.named_parameters())
    
    # First 10 layers
    first_10_layers = named_params[:5]
    
    # Last 10 layers
    last_10_layers = named_params[-5:]
    
    # Print the gradient norm for the first 10 layers
    print("First 5 Layers:")
    for name, param in first_10_layers:
        if param.grad is not None:
            print(f"Layer {name} | Grad Norm: {torch.norm(param.grad):.4f}")
    
    # Print the gradient norm for the last 10 layers
    print("Last 5 Layers:")
    for name, param in last_10_layers:
        if param.grad is not None:
            print(f"Layer {name} | Grad Norm: {torch.norm(param.grad):.4f}")

def aggregate_results(loader, model, device, dataset):
    """
    Computes `mu` for all samples in the base dataset and aggregates results.

    Args:
        base_dataset (Dataset): Original dataset.
        model (nn.Module): Model used to compute `mu`.
        device (torch.device): Device to run the computations on.

    Returns:
        data_tensor (torch.Tensor): Aggregated data tensor.
        mu_tensor (torch.Tensor): Aggregated `mu` tensor.
        target_tensor (torch.Tensor): Aggregated targets tensor.
    """
    model.eval()
    model.to(device)

    data_list, mu_right_list, mu_left_list, target_list = [], [], [], []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(loader, desc="Aggregating results")):
            # Move data to the appropriate device
            data = data.to(device)  # Add batch dimension
            if data.shape[0] != 32:
                print("data.shape[0] != 32")
                continue
            # Compute `mu`
            
            # my version
            right_limit, left_limit = model(data)  # Assuming model returns (mu, logvar, _)
            # lior version
            # mu, _, _ = model(data, dropout=True, mc_dropout=True, test=True)
            # mu = mu.clamp(0, 1).permute(1,0,2)
            # mu = mu.mean(dim=1)
            # if dataset =='oct':
            #     mu = torch.mean(mu, dim = 1, keepdim=True)
            #     target = torch.mean(target, dim=1, keepdim=True)

            # Store results
            data_list.append(data.cpu())
            mu_right_list.append(torch.abs(right_limit.cpu() - target.unsqueeze(-1).cpu()))
            mu_left_list.append(torch.abs(left_limit.cpu() - target.unsqueeze(-1).cpu()))
            target_list.append(target.cpu())




        
    # Stack all results into tensors
    data_tensor = torch.cat(data_list, dim=0)
    right_tensor = torch.cat(mu_right_list, dim=0).squeeze(0)
    target_tensor = torch.cat(target_list, dim=0).unsqueeze(-1)
    left_tensor = torch.cat(mu_left_list, dim=0).squeeze(0)
    if dataset =='oct':
        target_tensor = target_tensor.squeeze(-1)
    


    return data_tensor, right_tensor, left_tensor, target_tensor

class AggregatedDataset(Dataset):
    def __init__(self, data_tensor, mu_tensor, sd_tensor, target_tensor):
        """
        Args:
            data_tensor (torch.Tensor): Tensor containing data samples.
            mu_tensor (torch.Tensor): Tensor containing `mu` values.
            target_tensor (torch.Tensor): Tensor containing target values.
        """
        self.data_tensor = data_tensor
        self.mu_tensor = mu_tensor
        self.sd_tensor = sd_tensor
        self.target_tensor = target_tensor

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.mu_tensor[idx], self.sd_tensor[idx], self.target_tensor[idx]



class CustomMSELoss(nn.Module):
    def __init__(self, lambda_param=1.0):
        super(CustomMSELoss, self).__init__()
        self.lambda_param = lambda_param

    
    def forward(self, y_pred, y_true):
        distance_upper= torch.clamp(y_true - y_pred[:,0], min=0)
        distance_lower = torch.clamp(y_pred[:,1] - y_true, min=0)
        
        squared_distance_upper = distance_upper * distance_upper
        squared_distance_lower = distance_lower * distance_lower
        squared_distance = (squared_distance_upper.mean() + squared_distance_lower.mean()) * self.lambda_param
        
        in_interval = ((y_pred[:,0] >= y_true) & (y_pred[:,1] <= y_true )).float().mean()
        
        

        # combine mse loss and cross entropy loss
        mse_loss_upper = nn.functional.mse_loss(y_pred[:,0], y_true)
        mse_loss_lower = nn.functional.mse_loss(y_pred[:,1], y_true)
        
        mse_loss = mse_loss_upper + mse_loss_lower
        total_loss = mse_loss + squared_distance # +((1 - in_interval) ** 2)       
        return total_loss
    
def modify_predicted_distances(predicted_distances, probs):
    first_dim_results = (predicted_distances * probs).squeeze(-1)
    second_dim_results = (predicted_distances * (1 - probs)).squeeze(-1)
    return torch.stack([first_dim_results, second_dim_results], dim=1)

def sort_two_tensors(tensor1, tensor2):
    # Stack tensors along a new dimension
    stacked = torch.stack([tensor1, tensor2], dim=0)

    # Sort along that dimension
    sorted_values, _ = torch.sort(stacked, dim=0)

    # Extract max, middle, and min tensors
    min_values = sorted_values[0]  # Smallest values
    max_values = sorted_values[1]  # Largest values

    return max_values, min_values

def zero_smaller_pred(predicted_distances, scale_factor):
    # which distnce is bigger
    first_dim = predicted_distances[:, 0]
    second_dim = predicted_distances[:, 1]
    zero_first_dim = torch.where(first_dim * scale_factor < second_dim , torch.zeros_like(first_dim), first_dim)
    zero_second_dim = torch.where(second_dim * scale_factor< first_dim , torch.zeros_like(second_dim), second_dim)
    return torch.stack([zero_first_dim, zero_second_dim], dim=1)
    

def train(base_model= 'efficientnetb4',
          likelihood= 'gaussian',
          dataset = 'lumbar',
          dist_model_name = 'efficientnetb4',
          batch_size=32,
          init_lr=0.005,
          epochs=500,
          augment=True,
          valid_size=300,
          lr_patience=20,
          weight_decay=1e-8,
          lambda_param=1.0,
          scale_factor = 1,
          bigger = False,
          normalize = False,
          gpu=2,
          level=1):
    print("Current PID:", os.getpid())


    assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
    assert likelihood in ['gaussian', 'laplacian']
    assert dataset in ['breastpathq', 'boneage', 'endovis', 'oct', 'lumbar']
    assert gpu in [0, 1, 2,3]

    device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
    print("data_set =", dataset)
    print("model =", base_model)
    print("likelihood =", likelihood)
    print("batch_size =", batch_size)
    print("init_lr =", init_lr)
    print("epochs =", epochs)
    print("augment =", augment)
    print("valid_size =", valid_size)
    print("normalize", normalize)
    print("lr_patience =", lr_patience)
    print("weight_decay =", weight_decay)
    print("lambda param = ", lambda_param)
    print("bigget = ", bigger)
    print("scale_factor = ", scale_factor)
    print("device =", device)
    print("level =", level)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Print the time
    print("Current Time:", current_time)

    writer = SummaryWriter(comment=f"_{dataset}_{base_model}_{likelihood}")
    save_dir= '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots_new'

    dataset_name = dataset
    if dataset == 'lumbar':
        pred_x = False
        pred_y = False
        
        dataset_name = f'{dataset}_L{level}'
        data_set_train = LumbarDataset(level=level, mode='train', augment=True, scale=0.5, pred_x=pred_x, pred_y=pred_y)
        data_set_valid = LumbarDataset(level=level, mode='valid', augment=False, scale=0.5, pred_x=pred_x, pred_y=pred_y)
        model = load_trained_models.get_model_lumbar(base_model, level, None, device)
        model = DistNewModel(base_model, out_channels=1).to(device) 
        checkpoint = torch.load(f"/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots_new/efficientnetb4_lumbar_L{level}_snapshot_dist_efficientnetb4_lambda_1_scale_factor1_best.pth.tar", map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"epoch: {checkpoint['epoch']}")
        model.eval()    
        
        # dist_model = DistancePredictor(dist_model_name).to(device)
        dist_model = DistNewModel(base_model, in_channels=3, out_channels=1,
                             pretrained=True).to(device)
        if pred_x or pred_y:
            save_dir =f"{save_dir}/one_dim"
            if pred_x:
                save_dir= f"{save_dir}/x"
            else:
                save_dir = f"{save_dir}/y"        
    elif dataset=='boneage':
        resize_to = (256, 256)
        data_set_train = BoneAgeDataset(group='train', augment=augment, resize_to=resize_to)
        data_set_valid = BoneAgeDataset(augment=False, resize_to=resize_to,group='valid')
        
        model = load_trained_models.get_model_boneage(base_model, None, device)
        dist_model = DistancePredictor(dist_model_name, in_channels = 1).to(device)
    elif dataset == 'oct':
        data_set_train = OCTDataset(group='train')
        data_set_valid = OCTDataset(group='valid') 
        model = load_trained_models.get_model_oct(base_model, None, device)
        dist_model = DistancePredictor(dist_model_name, in_channels = 3).to(device)  
    else:
        assert False


    assert len(data_set_train) > 0
    assert len(data_set_valid) > 0

    print("len(data_set_train)", len(data_set_train))
    print("len(data_set_valid)", len(data_set_valid))

    train_loader = torch.utils.data.DataLoader(data_set_train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(data_set_valid, batch_size=batch_size, shuffle=False)


    dist_optimizer = optim.Adam(dist_model.parameters(), lr=init_lr, weight_decay=weight_decay)
    
    lr_scheduler_net = optim.lr_scheduler.ReduceLROnPlateau(dist_optimizer, patience=lr_patience, factor=0.1)

    loss_dist = nn.MSELoss()

    train_losses = []
    valid_losses = []
    dist_losses = []
    batch_counter = 0
    batch_counter_valid = 0

    
    data_tensor_train, mu_tensor_train, sd_tensor_train, target_tensor_train = aggregate_results(train_loader, model, device, dataset=dataset)
    data_tensor_valid, mu_tensor_valid, sd_tensor_valid, target_tensor_valid = aggregate_results(valid_loader, model, device, dataset=dataset)
    mu_tensor_train, sd_tensor_train = sort_two_tensors(mu_tensor_train, sd_tensor_train)
    mu_tensor_valid, sd_tensor_valid = sort_two_tensors(mu_tensor_valid, sd_tensor_valid)
    
    aggregated_dataset_train = AggregatedDataset(data_tensor_train, mu_tensor_train, sd_tensor_train, target_tensor_train)
    aggregated_dataset_valid = AggregatedDataset(data_tensor_valid, mu_tensor_valid, sd_tensor_valid, target_tensor_valid)
    
    train_loader = DataLoader(aggregated_dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(aggregated_dataset_valid, batch_size=batch_size, shuffle=True)

    
    
    try:
        for e in range(epochs):
            dist_model.train()

            epoch_train_loss = []
            dist_train_loss = []
            is_best = False

            for batch_idx, (data, right_d, left_d, targets) in enumerate(tqdm(train_loader)):
                data, right_d, left_d, target = data.to(device), right_d.to(device), left_d.to(device),  targets.to(device)
                
                if dataset =='boneage':
                    # data = data.repeat(1, 3, 1, 1)
                    targets = targets.squeeze(-1)

                # -------- Train Distance Predictor Model (predicting d+ and d-) --------
                dist_optimizer.zero_grad()

                # Forward pass for distance model
                predicted_distances = dist_model(data, dropout=True)
                # if normalize:
                #     true_d_plus = torch.clamp((targets - mu)/sd, min=0)
                #     true_d_minus = torch.clamp((mu - targets)/sd, min=0) 
                # else:
                #     true_d_plus = torch.max(targets, mu)
                #     true_d_minus = torch.min(targets, mu)
                    # true_d_plus = torch.clamp(targets - mu, min=0) 
                    # true_d_minus = torch.clamp(mu - targets, min=0)
                # true_distances = torch.stack([true_d_plus, true_d_minus], dim=1).squeeze(-1)
                predicted_distances = torch.stack([predicted_distances[0], predicted_distances[1]], dim=1).squeeze(-1)
                true_distance = torch.stack([right_d.squeeze(-1), left_d.squeeze(-1)], dim=1).squeeze(-1)
                # Compute loss for distance model

                dist_loss = loss_dist(predicted_distances.float(), true_distance.float())

                # Backward pass for the combined loss
                dist_loss.backward()
                # print_grad_norms(dist_model=dist_model)
                dist_optimizer.step()
                


                # Track training metrics for distance model
                dist_train_loss.append(dist_loss.item())
                writer.add_scalar('dist_train/loss', dist_loss.item(), batch_counter)

                batch_counter += 1
                

            avg_dist_loss = sum(dist_train_loss) / len(dist_train_loss)
            print(f"Epoch {e+1}/{epochs} -  Distance Loss: {avg_dist_loss:.4f}")
            lr_scheduler_net.step(avg_dist_loss)



            # model.eval()
            dist_model.eval()

            epoch_valid_loss = []
            dist_valid_loss = []  # To store distance model losses
            targets_valid = []

            with torch.no_grad():
                for batch_idx, (data, right_d, left_d, targets) in enumerate(tqdm(valid_loader)):
                    data, right_d, left_d, targets = data.to(device), right_d.to(device), left_d.to(device),  targets.to(device)

                    if dataset =='boneage':
                        # data = data.repeat(1, 3, 1, 1)
                        targets = targets.squeeze(-1)

                    targets_valid.append(targets.detach().cpu())


                    # -------- Evaluate Distance Model --------
                    predicted_distances = dist_model(data, dropout=True) # Predict d+ and d-
                    # if normalize:
                    #     true_d_plus = torch.clamp((targets - mu)/sd, min=0)
                    #     true_d_minus = torch.clamp((mu - targets)/sd, min=0) 
                    # else:
                    #     true_d_plus = torch.max(targets, mu)
                    #     true_d_minus = torch.min(targets, mu)
                        # true_d_plus = torch.clamp(targets - mu, min=0)
                        # true_d_minus = torch.clamp(mu - targets, min=0) 
                    predicted_distances = torch.stack([predicted_distances[0], predicted_distances[1]], dim=1).squeeze(-1)
                    true_distances = torch.stack([right_d.squeeze(-1), left_d.squeeze(-1)], dim=1).squeeze(-1)                    # print("true distance:", true_distances[:1])
                    
                    # print("Predicted distances:", predicted_distances[:1])
                    dist_loss = loss_dist(predicted_distances.float(), true_distances.float())
                    dist_valid_loss.append(dist_loss.item())
                    

                    # writer.add_scalar('dist_valid/loss', dist_loss.item(), batch_counter_valid)

                    batch_counter_valid += 1
                    

            # Compute metrics for the epoch
            epoch_dist_valid_loss = np.mean(dist_valid_loss)
            targets_valid = torch.cat(targets_valid, dim=0)

            print(f"Epoch {e}:")
            print(f"dist_loss: {epoch_dist_valid_loss:.5f}")

            # Save epoch losses
            train_losses.append(epoch_train_loss)
            valid_losses.append(epoch_valid_loss)
            dist_losses.append(epoch_dist_valid_loss)  # Store distance model's validation loss

            if dist_losses[-1] <= np.min(dist_losses):
                save_snapshot(save_dir, dist_model_name, dataset_name, e, dist_model, base_model, lambda_param=lambda_param, is_mornalize=normalize, scale_factor=scale_factor, is_best=True) 


            save_snapshot(save_dir, dist_model_name, dataset_name, e, dist_model, base_model, lambda_param=lambda_param, is_mornalize=normalize, scale_factor=scale_factor) 
    except KeyboardInterrupt:
            save_snapshot(save_dir, dist_model_name, dataset_name, e, dist_model, base_model, lambda_param=lambda_param, is_mornalize=normalize,  scale_factor=scale_factor)
            
            
if __name__ == '__main__':
    fire.Fire(train)