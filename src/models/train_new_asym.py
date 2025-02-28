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
from models import BreastPathQModel, DistancePredictor, DistNewModel, BreastPathQModel3Heads
from utils import kaiming_normal_init
from utils import nll_criterion_gaussian, nll_criterion_laplacian
import torch.nn as nn
import load_trained_models 
from torch.utils.data import Dataset, DataLoader


def save_snapshot(save_dir, model_name, dataset_name, epoch, model,  is_best = False):
    suffix = 'best' if is_best else 'new'
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/{model_name}_{dataset_name}_snapshot_{suffix}.pth.tar"
    print(f"Saving snapshot, path: {filename}")
    torch.save({
        'epoch': epoch,
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






class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()


    
    def forward(self, y_pred, y_true):
        mu_preds = y_pred[:, 0]
        sigma_left_preds = y_pred[:, 1]
        sigma_right_preds = y_pred[:, 2]
        
        # left_variance = torch.exp(sigma_left_preds)
        # right_variance = torch.exp(sigma_right_preds)
        
        # log_left_var = torch.log(left_variance)
        # log_right_var = torch.log(right_variance)
        
        log_left_var = sigma_left_preds
        log_right_var = sigma_right_preds
        
        preds_bigger = mu_preds > y_true
        preds_smaller = mu_preds <= y_true
        
        loss_bigger = self._calc_loss(mu_preds, log_right_var, y_true, preds_bigger)
        loss_smaller = self._calc_loss(mu_preds, log_left_var, y_true, preds_smaller)
        total_loss = loss_bigger + loss_smaller
        print(f"loss_bigger: {loss_bigger.item()} | loss_smaller: {loss_smaller.item()}")
        return total_loss
        
    
    def _calc_loss(self, mu, log_var, target, filter_condition, reduction='mean'):
        mu_filtered = mu[filter_condition]
        log_var_filtered = log_var[filter_condition]
        target_filtered = target[filter_condition]
        if len(mu_filtered) == 0:
            return torch.tensor(0.0, device=mu.device)
        log_var_filtered = torch.clamp(log_var_filtered, min=-10, max=10)
        return nll_criterion_gaussian(mu_filtered.unsqueeze(-1), log_var_filtered.unsqueeze(-1), target_filtered.unsqueeze(-1), reduction=reduction).to(mu.device) + 1e-8
        
        





    

def train(base_model= 'densenet201',
          likelihood= 'gaussian',
          dataset = 'lumbar',
          batch_size=32,
          init_lr=0.001,
          epochs=52,
          augment=True,
          valid_size=300,
          lr_patience=20,
          weight_decay=1e-8,
          gpu=1,
          level=2):
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
    print("lr_patience =", lr_patience)
    print("weight_decay =", weight_decay)
    print("device =", device)
    print("level =", level)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Print the time
    print("Current Time:", current_time)

    writer = SummaryWriter(comment=f"_{dataset}_{base_model}_{likelihood}")
    save_dir= '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots_asym'

    dataset_name = dataset
    if dataset == 'lumbar':
        pred_x = False
        pred_y = False
        
        dataset_name = f'{dataset}_L{level}'
        data_set_train = LumbarDataset(level=level, mode='train', augment=True, scale=0.5, pred_x=pred_x, pred_y=pred_y)
        data_set_valid = LumbarDataset(level=level, mode='valid', augment=False, scale=0.5, pred_x=pred_x, pred_y=pred_y)
        dist_model = BreastPathQModel3Heads(base_model, in_channels=3, out_channels=1,
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
        
        dist_model = DistancePredictor(base_model, in_channels = 1).to(device)
    elif dataset == 'oct':
        data_set_train = OCTDataset(group='train')
        data_set_valid = OCTDataset(group='valid') 
        dist_model = DistancePredictor(base_model, in_channels = 3).to(device)  
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

    loss_dist = CustomLoss()

    train_losses = []
    valid_losses = []
    dist_losses = []
    batch_counter = 0
    batch_counter_valid = 0

    

    
    
    try:
        for e in range(epochs):
            dist_model.train()

            epoch_train_loss = []
            dist_train_loss = []
            is_best = False

            for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
                data, targets = data.to(device),  targets.to(device)
                
                if dataset =='boneage':
                    # data = data.repeat(1, 3, 1, 1)
                    targets = targets.squeeze(-1)

                # -------- Train Distance Predictor Model (predicting d+ and d-) --------
                dist_optimizer.zero_grad()

                # Forward pass for distance model
                predictions = dist_model(data, dropout=True)
                predictions = torch.stack([predictions[0], predictions[1], predictions[2]], dim=1).squeeze(-1)
                predictions = torch.clamp(predictions, min=-10, max=10)

                # Compute loss for distance model

                dist_loss = loss_dist(predictions.float(), targets.float())

                # Backward pass for the combined loss
                dist_loss.backward()
                torch.nn.utils.clip_grad_norm_(dist_model.parameters(), max_norm=1.0)
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
                for batch_idx, (data, targets) in enumerate(tqdm(valid_loader)):
                    data,  targets = data.to(device), targets.to(device)

                    if dataset =='boneage':
                        # data = data.repeat(1, 3, 1, 1)
                        targets = targets.squeeze(-1)

                    targets_valid.append(targets.detach().cpu())


                    # -------- Evaluate Distance Model --------
                    predictions = dist_model(data, dropout=True) # Predict d+ and d-
                    predictions = torch.stack([predictions[0], predictions[1], predictions[2]], dim=1).squeeze(-1)

                    dist_loss = loss_dist(predictions.float(), targets.float())
                    dist_valid_loss.append(dist_loss.item())
                    


                    batch_counter_valid += 1
                    
            torch.cuda.empty_cache()
            
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
                save_snapshot(save_dir, base_model, dataset_name, e, dist_model, is_best=True) 


            save_snapshot(save_dir, base_model, dataset_name, e, dist_model) 
    except KeyboardInterrupt:
            save_snapshot(save_dir, base_model, dataset_name, e, dist_model )
            
            
if __name__ == '__main__':
    fire.Fire(train)