# Max-Heinrich Laves
# Institute of Mechatronic Systems
# Leibniz Universität Hannover, Germany
# 2019

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
from models import BreastPathQModel, DistancePredictor
from utils import kaiming_normal_init
from utils import nll_criterion_gaussian, nll_criterion_laplacian
import torch.nn as nn


def save_snapshot(model_name, likelihood, dataset_name, epoch, model, optimizer, train_losses, valid_losses, dist_losses, best_val_loss, dist_base_model= False):
    os.makedirs('/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots_new', exist_ok=True)
    dist_str = f'dist_{dist_base_model}' if dist_base_model is not None else ''
    filename = f"/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots_new/{model_name}_{likelihood}_{dataset_name}_snapshot_{dist_str}.pth.tar"
    print(f"Saving snapshot at epoch {epoch} with val_loss: {best_val_loss:.5f}, path: {filename}")
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': valid_losses,
        'dist_losses': dist_losses,
    }, filename)    

torch.backends.cudnn.benchmark = True


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
          dataset = 'lumbar',
         dist_model_name = 'resnet50',
          batch_size=32,
          init_lr=0.001,
          epochs=50,
          augment=True,
          valid_size=300,
          lr_patience=20,
          weight_decay=1e-8,
          gpu=0,
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
    print("device =", device)

    writer = SummaryWriter(comment=f"_{dataset}_{base_model}_{likelihood}")


    dataset_name = dataset
    if dataset == 'lumbar':
        dataset_name = f'{dataset}_L{level}'

    if dataset == 'lumbar':
        in_channels = 3
        out_channels = 1
        pretrained = True

        

        data_set_train = LumbarDataset(level=level, mode='train', augment=True, scale=0.5)
        data_set_valid = LumbarDataset(level=level, mode='valid', augment=False, scale=0.5)

        assert len(data_set_train) > 0
        assert len(data_set_valid) > 0

        print("len(data_set_train)", len(data_set_train))
        print("len(data_set_valid)", len(data_set_valid))

        train_loader = torch.utils.data.DataLoader(data_set_train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(data_set_valid, batch_size=batch_size, shuffle=True)
    else:
        assert False

    model = BreastPathQModel(base_model, in_channels=in_channels, out_channels=out_channels,
                             pretrained=pretrained).to(device)
    dist_model = DistancePredictor(dist_model_name).to(device)
    dist_optimizer = optim.Adam(dist_model.parameters(), lr=1e-3)
    loss_dist = CustomMSELoss(lambda_param=5)

    if not pretrained:
        kaiming_normal_init(model)

    if likelihood == 'gaussian':
        nll_criterion = nll_criterion_gaussian
        metric = torch.nn.functional.mse_loss
    elif likelihood == 'laplacian':
        nll_criterion = nll_criterion_laplacian
        metric = torch.nn.functional.l1_loss
    else:
        assert False

    if dataset == 'breastpathq' or 'boneage':
        optimizer_net = optim.SGD(model.parameters(), lr=init_lr, weight_decay=weight_decay, momentum=0.9)
        print("SGD(model.parameters(), lr=init_lr, weight_decay=weight_decay, momentum=0.9)")
    else:
        optimizer_net = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
        print("AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)")

    print("ReduceLROnPlateau(optimizer_net, patience=lr_patience, factor=0.1)")
    lr_scheduler_net = optim.lr_scheduler.ReduceLROnPlateau(optimizer_net, patience=lr_patience, factor=0.1)


    train_losses = []
    valid_losses = []
    dist_losses = []
    batch_counter = 0
    batch_counter_valid = 0

    try:
        for e in range(epochs):
            model.train()
            dist_model.train()

            epoch_train_loss = []
            dist_train_loss = []
            mu_train = []
            logvar_train = []
            targets_train = []
            is_best = False

            print("lr =", optimizer_net.param_groups[0]['lr'])

            for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
                data, targets = data.to(device), targets.to(device)
                targets = targets.unsqueeze(-1)

                # -------- Train Basic Model (predicting y) --------
                optimizer_net.zero_grad()
                mu, logvar, _ = model(data, dropout=True)
                loss = nll_criterion(mu, logvar, targets).to(device)
                # loss.backward()
                # optimizer_net.step()

                # Track training metrics for basic model
                epoch_train_loss.append(loss.item())
                targets_train.append(targets.detach().cpu())
                mu_train.append(mu.detach().cpu())
                logvar_train.append(logvar.detach().cpu())

                writer.add_scalar('train/loss', loss.item(), batch_counter)
                writer.add_scalar('train/mse', metric(mu, targets), batch_counter)
                writer.add_scalar('train/var', logvar.exp().mean(), batch_counter)

                # -------- Train Distance Predictor Model (predicting d+ and d-) --------
                dist_optimizer.zero_grad()

                # Forward pass for distance model
                predicted_distances = dist_model(data)
                true_d_plus = torch.clamp(targets - mu, min=0) * 3.0 # True d+
                true_d_minus = torch.clamp(mu - targets, min=0) * 3.0 # True d-
                true_distances = torch.stack([true_d_plus, true_d_minus], dim=1).squeeze(-1)

                # Compute loss for distance model
                dist_loss = loss_dist(predicted_distances.float(), true_distances.float())
                # dist_loss.backward(retain_graph=True)
                # dist_optimizer.step()
                total_loss = loss + dist_loss  # Add primary model loss and distance model loss

                # Backward pass for the combined loss
                total_loss.backward()
                # for name, param in dist_model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Layer {name} | Grad Norm: {torch.norm(param.grad):.4f}")
                dist_optimizer.step()
                optimizer_net.step()
                


                # Track training metrics for distance model
                dist_train_loss.append(dist_loss.item())
                writer.add_scalar('dist_train/loss', dist_loss.item(), batch_counter)

                batch_counter += 1
                
                

            # Log overall epoch metrics
            avg_loss = sum(epoch_train_loss) / len(epoch_train_loss)
            avg_dist_loss = sum(dist_train_loss) / len(dist_train_loss)
            print(f"Epoch {e+1}/{epochs} - Loss: {avg_loss:.4f}, Distance Loss: {avg_dist_loss:.4f}")

            # TODO: can be modified
            epoch_train_loss = np.mean(epoch_train_loss)
            lr_scheduler_net.step(epoch_train_loss)

            targets_train = torch.cat(targets_train, dim=0)
            mu_train = torch.cat(mu_train, dim=0)
            logvar_train = torch.cat(logvar_train, dim=0)
            # mse_train = metric(mu_train, targets_train)
            mse_train = metric(mu_train, targets_train)


            model.eval()
            dist_model.eval()

            epoch_valid_loss = []
            dist_valid_loss = []  # To store distance model losses
            mu_valid = []
            logvar_valid = []
            targets_valid = []

            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(tqdm(valid_loader)):
                    data, targets = data.to(device), targets.to(device)
                    targets = targets.unsqueeze(-1)

                    # -------- Evaluate Basic Model --------
                    mu, logvar, _ = model(data, dropout=True)
                    loss = nll_criterion(mu, logvar, targets).to(device)
                    epoch_valid_loss.append(loss.item())

                    targets_valid.append(targets.detach().cpu())
                    mu_valid.append(mu.detach().cpu())
                    logvar_valid.append(logvar.detach().cpu())

                    writer.add_scalar('valid/loss', loss.item(), batch_counter_valid)
                    writer.add_scalar('valid/mse', metric(mu, targets), batch_counter_valid)
                    writer.add_scalar('valid/var', logvar.exp().mean(), batch_counter_valid)

                    # -------- Evaluate Distance Model --------
                    predicted_distances = dist_model(data) / 3.0 # Predict d+ and d-
                    true_d_plus = torch.clamp(targets - mu, min=0)  # True d+
                    true_d_minus = torch.clamp(mu - targets, min=0)  # True d-
                    true_distances = torch.stack([true_d_plus, true_d_minus], dim=1).squeeze(-1)

                    dist_loss = nn.functional.mse_loss(predicted_distances.float(), true_distances.float())
                    dist_valid_loss.append(dist_loss.item())

                    writer.add_scalar('dist_valid/loss', dist_loss.item(), batch_counter_valid)

                    batch_counter_valid += 1
                    

            # Compute metrics for the epoch
            epoch_valid_loss = np.mean(epoch_valid_loss)
            epoch_dist_valid_loss = np.mean(dist_valid_loss)
            targets_valid = torch.cat(targets_valid, dim=0)
            mu_valid = torch.cat(mu_valid, dim=0)
            logvar_valid = torch.cat(logvar_valid, dim=0)
            mse_valid = metric(mu_valid, targets_valid)

            print(f"Epoch {e}:")
            print(f"train: loss: {epoch_train_loss:.5f}, mse: {mse_train:.5f}, var: {logvar_train.exp().mean():.5f}")
            print(f"valid: loss: {epoch_valid_loss:.5f}, mse: {mse_valid:.5f}, var: {logvar_valid.exp().mean():.5f}, dist_loss: {epoch_dist_valid_loss:.5f}")

            # Save epoch losses
            train_losses.append(epoch_train_loss)
            valid_losses.append(epoch_valid_loss)
            dist_losses.append(epoch_dist_valid_loss)  # Store distance model's validation loss

            if valid_losses[-1] <= np.min(valid_losses):
                is_best = True


            # if is_best:
            #     os.makedirs('./snapshots', exist_ok=True)
            #     filename = f"./snapshots/{base_model}_{likelihood}_{dataset_name}_best_trans.pth.tar"
            #     print(f"Saving best weights so far with val_loss: {valid_losses[-1]:.5f}")
            #     torch.save({
            #         'epoch': e,
            #         'state_dict': model.state_dict(),
            #         'optimizer': optimizer_net.state_dict(),
            #         'train_losses': train_losses,
            #         'val_losses': valid_losses,
            #     }, filename)

            if optimizer_net.param_groups[0]['lr'] < 1e-7:
                break

            save_snapshot(base_model, likelihood, dataset_name, e, model, optimizer_net, train_losses, valid_losses, dist_losses, valid_losses[-1], dist_base_model=None)
            save_snapshot(dist_model_name, likelihood, dataset_name, e, dist_model, dist_optimizer, train_losses, valid_losses, dist_losses, valid_losses[-1], dist_base_model=base_model)

    except KeyboardInterrupt:
        save_snapshot(base_model, likelihood, dataset_name, e, model, optimizer_net, train_losses, valid_losses, dist_losses, valid_losses[-1], dist_base_model=None)
        save_snapshot(dist_model_name, likelihood, dataset_name, e, dist_model, dist_optimizer, train_losses, valid_losses, dist_losses, valid_losses[-1], dist_base_model=base_model)


if __name__ == '__main__':
    fire.Fire(train)
