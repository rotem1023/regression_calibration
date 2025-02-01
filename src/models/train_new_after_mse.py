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
from data_generator_endovis import EndoVisDataset
from data_generator_lumbar import LumbarDataset
from data_generator_oct import OCTDataset
from models import BreastPathQModelOneOutput, DistancePredictor
from utils import kaiming_normal_init
from utils import nll_criterion_gaussian, nll_criterion_laplacian
import torch.nn as nn
import load_trained_models 
from torch.utils.data import Dataset, DataLoader


def save_snapshot(model_name, dataset_name, epoch, model, dist_base_model_name,lambda_param, is_best = False):
    suffix = 'best' if is_best else 'new'
    os.makedirs('/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots_new_mse', exist_ok=True)
    dist_str = f'dist_{dist_base_model_name}' if dist_base_model_name is not None else ''
    filename = f"/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots_new_mse/{model_name}_{dataset_name}_snapshot_{dist_str}_lambda_{int(lambda_param)}_{suffix}.pth.tar"
    print(f"Saving snapshot, path: {filename}")
    torch.save({
        'epoch': epoch,
        'lambda_param': lambda_param,
        'state_dict': model.state_dict(),
    }, filename)    

torch.backends.cudnn.benchmark = True


def aggregate_results(base_dataset, model, device):
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

    data_list, mu_list, target_list = [], [], []
    with torch.no_grad():
        for data, target in tqdm(base_dataset, desc="Aggregating results"):
            # Move data to the appropriate device
            data = data.to(device)  # Add batch dimension
            if data.shape[0] != 32:
                print("data.shape[0] != 32")
                continue
            # Compute `mu`
            mu = model(data)  # Assuming model returns (mu, logvar, _)


            # Store results
            data_list.append(data.cpu())
            mu_list.append(mu.cpu())
            target_list.append(target.cpu())


        

    # Stack all results into tensors
    data_tensor = torch.cat(data_list, dim=0)
    mu_tensor = torch.cat(mu_list, dim=0).squeeze(0)
    target_tensor = torch.cat(target_list, dim=0).unsqueeze(-1)

    return data_tensor, mu_tensor, target_tensor

class AggregatedDataset(Dataset):
    def __init__(self, data_tensor, mu_tensor, target_tensor):
        """
        Args:
            data_tensor (torch.Tensor): Tensor containing data samples.
            mu_tensor (torch.Tensor): Tensor containing `mu` values.
            target_tensor (torch.Tensor): Tensor containing target values.
        """
        self.data_tensor = data_tensor
        self.mu_tensor = mu_tensor
        self.target_tensor = target_tensor

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.mu_tensor[idx], self.target_tensor[idx]



class CustomMSELoss(nn.Module):
    def __init__(self, lambda_param=1.0):
        super(CustomMSELoss, self).__init__()
        self.lambda_param = lambda_param
    
    def forward(self, y_pred, y_true):
        # Calculate the difference between predicted and true values
        diff = y_pred - y_true
        
        # Penalize predictions smaller than the true values
        loss_smaller = torch.where(diff < 0, self.lambda_param * torch.square(diff), torch.zeros_like(diff))
        
        # Penalize predictions larger than the true values with a regular penalty
        loss_larger = torch.where(diff > 0, torch.square(diff), torch.zeros_like(diff))
        
        # Combine both penalties
        total_loss = torch.mean(loss_smaller + loss_larger)
                
        return total_loss

def train(base_model= 'densenet201',
          likelihood= 'gaussian',
          dataset = 'boneage',
         dist_model_name = 'resnet50',
          batch_size=32,
          init_lr=0.001,
          epochs=50,
          augment=True,
          valid_size=300,
          lr_patience=20,
          weight_decay=1e-8,
          lambda_param=1.0,
          gpu=1,
          level=5):
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
    print("lambda param = ", lambda_param)
    print("device =", device)
    print("level =", level)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Print the time
    print("Current Time:", current_time)

    writer = SummaryWriter(comment=f"_{dataset}_{base_model}_{likelihood}")
    loss = "mse"


    dataset_name = dataset
    if dataset == 'lumbar':
        dataset_name = f'{dataset}_L{level}'
        data_set_train = LumbarDataset(level=level, mode='train', augment=True, scale=0.5)
        data_set_valid = LumbarDataset(level=level, mode='valid', augment=False, scale=0.5)
        model = load_trained_models.get_model_lumbar(base_model, level, None, device, loss = loss)
        dist_model = DistancePredictor(dist_model_name).to(device)
    elif dataset=='boneage':
        resize_to = (256, 256)
        data_set_train = BoneAgeDataset(group='train', augment=augment, resize_to=resize_to)
        data_set_valid = BoneAgeDataset(augment=False, resize_to=resize_to,group='valid')
        
        model = load_trained_models.get_model_boneage(base_model, None, device, loss=loss)
        dist_model = DistancePredictor(dist_model_name, in_channels = 1).to(device)

    else:
        assert False


    assert len(data_set_train) > 0
    assert len(data_set_valid) > 0

    print("len(data_set_train)", len(data_set_train))
    print("len(data_set_valid)", len(data_set_valid))

    train_loader = torch.utils.data.DataLoader(data_set_train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(data_set_valid, batch_size=batch_size, shuffle=True)


    dist_optimizer = optim.Adam(dist_model.parameters(), lr=1e-3)
    loss_dist = CustomMSELoss(lambda_param=lambda_param)


    train_losses = []
    valid_losses = []
    dist_losses = []
    batch_counter = 0
    batch_counter_valid = 0

    
    data_tensor_train, mu_tensor_train, target_tensor_train = aggregate_results(train_loader, model, device)
    data_tensor_valid, mu_tensor_valid, target_tensor_valid = aggregate_results(valid_loader, model, device)
    aggregated_dataset_train = AggregatedDataset(data_tensor_train, mu_tensor_train, target_tensor_train)
    aggregated_dataset_valid = AggregatedDataset(data_tensor_valid, mu_tensor_valid, target_tensor_valid)
    
    train_loader = DataLoader(aggregated_dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(aggregated_dataset_valid, batch_size=batch_size, shuffle=True)

    
    
    try:
        for e in range(epochs):
            dist_model.train()

            epoch_train_loss = []
            dist_train_loss = []
            is_best = False

            for batch_idx, (data, mu,targets) in enumerate(tqdm(train_loader)):
                data, mu, targets = data.to(device), mu.to(device), targets.to(device)
                
                if dataset =='boneage':
                    targets = targets.squeeze(-1)

                # -------- Train Distance Predictor Model (predicting d+ and d-) --------
                dist_optimizer.zero_grad()

                # Forward pass for distance model
                predicted_distances = dist_model(data)
                true_d_plus = torch.clamp(targets - mu, min=0) # True d+
                true_d_minus = torch.clamp(mu - targets, min=0) # True d-
                true_distances = torch.stack([true_d_plus, true_d_minus], dim=1).squeeze(-1)

                # Compute loss for distance model
                dist_loss = loss_dist(predicted_distances.float(), true_distances.float())
                # dist_loss.backward(retain_graph=True)
                # dist_optimizer.step()

                # Backward pass for the combined loss
                dist_loss.backward()
                # for name, param in dist_model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Layer {name} | Grad Norm: {torch.norm(param.grad):.4f}")
                dist_optimizer.step()
                


                # Track training metrics for distance model
                dist_train_loss.append(dist_loss.item())
                writer.add_scalar('dist_train/loss', dist_loss.item(), batch_counter)

                batch_counter += 1
                
    
            avg_dist_loss = sum(dist_train_loss) / len(dist_train_loss)
            print(f"Epoch {e+1}/{epochs} -  Distance Loss: {avg_dist_loss:.4f}")




            model.eval()
            dist_model.eval()

            dist_valid_loss = []  # To store distance model losses
            targets_valid = []

            with torch.no_grad():
                for batch_idx, (data, mu, targets) in enumerate(tqdm(valid_loader)):
                    data, mu,  targets = data.to(device), mu.to(device), targets.to(device)

                    if dataset =='boneage':
                        targets = targets.squeeze(-1)

                    targets_valid.append(targets.detach().cpu())


                    # -------- Evaluate Distance Model --------
                    predicted_distances = dist_model(data) # Predict d+ and d-
                    true_d_plus = torch.clamp(targets - mu, min=0)  # True d+
                    true_d_minus = torch.clamp(mu - targets, min=0)  # True d-
                    true_distances = torch.stack([true_d_plus, true_d_minus], dim=1).squeeze(-1)

                    dist_loss = nn.functional.mse_loss(predicted_distances.float(), true_distances.float())
                    dist_valid_loss.append(dist_loss.item())

                    writer.add_scalar('dist_valid/loss', dist_loss.item(), batch_counter_valid)

                    batch_counter_valid += 1
                    

            # Compute metrics for the epoch
            epoch_dist_valid_loss = np.mean(dist_valid_loss)
            targets_valid = torch.cat(targets_valid, dim=0)

            print(f"Epoch {e}:")
            print(f"valid: dist_loss: {epoch_dist_valid_loss:.5f}")

            # Save epoch losses
            train_losses.append(epoch_train_loss)
            dist_losses.append(epoch_dist_valid_loss)  # Store distance model's validation loss

            if valid_losses[-1] <= np.min(valid_losses):
                save_snapshot(dist_model_name, dataset_name, e, dist_model, base_model,lambda_param=lambda_param, is_best=True)


            save_snapshot(dist_model_name, dataset_name, e, dist_model, base_model, lambda_param=lambda_param)
    except KeyboardInterrupt:
            save_snapshot(dist_model_name, dataset_name, e, dist_model, base_model, lambda_param=lambda_param)
            
            
if __name__ == '__main__':
    fire.Fire(train)
