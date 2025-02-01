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
from models import BreastPathQModelOneOutput
from utils import kaiming_normal_init
from utils import nll_criterion_gaussian, nll_criterion_laplacian
from utils import save_current_snapshot


torch.backends.cudnn.benchmark = True


def train(base_model= 'densenet201',
          dataset = 'boneage',
          batch_size=32,
          init_lr=0.001,
          epochs=100,
          augment=True,
          valid_size=300,
          lr_patience=20,
          weight_decay=1e-8,
          gpu=0,
          level=5):
    print("Current PID:", os.getpid())

    likelihood = 'mse'
    assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
    assert likelihood in ['mse']
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
    print(f"level: {level}")

    writer = SummaryWriter(comment=f"_{dataset}_{base_model}_{likelihood}")

    resize_to = (256, 256)
    dataset_name = dataset
    if dataset == 'lumbar':
        dataset_name = f'{dataset}_L{level}'

    if dataset == 'boneage':
        in_channels = 1
        out_channels = 1
        pretrained = False

        data_set_train = BoneAgeDataset(group='train', augment=False, resize_to=resize_to)
        data_set_valid = BoneAgeDataset(augment=False, resize_to=resize_to,group='valid')
  
        # assert len(data_set_train) > 0
        # assert len(data_set_valid) > 0

        train_loader = torch.utils.data.DataLoader(data_set_train, batch_size=32, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(data_set_valid, batch_size=32, shuffle=True)
    elif dataset == 'lumbar':
        in_channels = 3
        out_channels = 1
        pretrained = True

        

        data_set_train = LumbarDataset(level=level, mode='train', augment=True, scale=0.5)
        data_set_valid = LumbarDataset(level=level, mode='valid', augment=False, scale=0.5)

        assert len(data_set_train) > 0
        assert len(data_set_valid) > 0

        print("len(data_set_train)", len(data_set_train))
        print("len(data_set_valid)", len(data_set_valid))

        train_loader = torch.utils.data.DataLoader(data_set_train, batch_size=32, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(data_set_valid, batch_size=32, shuffle=True)
    elif dataset == 'oct':
        in_channels = 3
        out_channels = 6
        pretrained = True

        data_dir = '/media/fastdata/laves/oct_data_needle/data'

        data_set_train = OCTDataset(data_dir=data_dir, augment=True, resize_to=resize_to, preload=True)
        data_set_valid = OCTDataset(data_dir=data_dir, augment=False, preloaded_data_from=data_set_train)

        assert len(data_set_train) > 0
        assert len(data_set_valid) > 0

        # indices = torch.randperm(len(data_set_valid))
        # train_indices = indices[:len(indices) - 2*valid_size]
        # valid_indices = indices[len(indices) - 2*valid_size:len(indices) - 1*valid_size]
        # test_indices = indices[len(indices) - 1*valid_size:]
        # torch.save(train_indices, f'./{dataset}_train_indices.pth')
        # torch.save(valid_indices, f'./{dataset}_valid_indices.pth')
        # torch.save(test_indices, f'./{dataset}_test_indices.pth')

        train_indices = torch.load(f'./data_indices/{dataset}_train_indices.pth')
        valid_indices = torch.load(f'./data_indices/{dataset}_valid_indices.pth')

        train_loader = torch.utils.data.DataLoader(data_set_train, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(train_indices))
        valid_loader = torch.utils.data.DataLoader(data_set_valid, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(valid_indices))
    else:
        assert False
    model = BreastPathQModelOneOutput(base_model, in_channels=in_channels, out_channels=out_channels,
                             pretrained=pretrained).to(device)
    
    


    nll_criterion = torch.nn.functional.mse_loss
    metric = torch.nn.functional.mse_loss


    if dataset == 'breastpathq' or 'boneage':
        optimizer_net = optim.SGD(model.parameters(), lr=init_lr, weight_decay=weight_decay, momentum=0.9)
        print("SGD(model.parameters(), lr=init_lr, weight_decay=weight_decay, momentum=0.9)")
    else:
        optimizer_net = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
        print("AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)")

    print("ReduceLROnPlateau(optimizer_net, patience=lr_patience, factor=0.1)")
    lr_scheduler_net = optim.lr_scheduler.ReduceLROnPlateau(optimizer_net, patience=lr_patience, factor=0.1)

    print("")

    train_losses = []
    valid_losses = []
    batch_counter = 0
    batch_counter_valid = 0

    try:
        for e in range(epochs):
            model.train()

            epoch_train_loss = []
            mu_train = []
            targets_train = []
            is_best = False

            print("lr =", optimizer_net.param_groups[0]['lr'])
            for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
                data, targets = data.to(device), targets.to(device)
                optimizer_net.zero_grad()
                mu = model(data, dropout=True)
                loss = nll_criterion(mu, targets).to(device)
                loss.backward()
                epoch_train_loss.append(loss.item())
                optimizer_net.step()

                targets_train.append(targets.detach().cpu())
                mu_train.append(mu.detach().cpu())

                writer.add_scalar('train/loss', loss.item(), batch_counter)
                writer.add_scalar('train/mse', metric(mu, targets), batch_counter)
                batch_counter += 1
                
                if torch.isnan(mu).any().item():
                    print("None in mu, epoch:", e)
                    return

                

            epoch_train_loss = np.mean(epoch_train_loss)
            lr_scheduler_net.step(epoch_train_loss)

            targets_train = torch.cat(targets_train, dim=0)
            mu_train = torch.cat(mu_train, dim=0)
            mse_train = metric(mu_train, targets_train)


            model.eval()
            epoch_valid_loss = []
            mu_valid = []
            targets_valid = []

            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(tqdm(valid_loader)):
                    data, targets = data.to(device), targets.to(device)
                    mu = model(data, dropout=True)
                    loss = nll_criterion(mu,targets).to(device)
                    epoch_valid_loss.append(loss.item())

                    targets_valid.append(targets.detach().cpu())
                    mu_valid.append(mu.detach().cpu())

                    writer.add_scalar('valid/loss', loss.item(), batch_counter_valid)
                    writer.add_scalar('valid/mse', metric(mu, targets), batch_counter_valid)
                    batch_counter_valid += 1

                    

            epoch_valid_loss = np.mean(epoch_valid_loss)
            targets_valid = torch.cat(targets_valid, dim=0)
            mu_valid = torch.cat(mu_valid, dim=0)
            mse_valid = metric(mu_valid, targets_valid)

            print(f"Epoch {e}:")
            print(f"train: loss: {epoch_train_loss:.5f}, mse: {mse_train:.5f}")
            print(f"valid: loss: {epoch_valid_loss:.5f}, mse: {mse_valid:.5f}")

            # save epoch losses
            train_losses.append(epoch_train_loss)
            valid_losses.append(epoch_valid_loss)

            if valid_losses[-1] <= np.min(valid_losses):
                is_best = True

            if is_best:
                os.makedirs('./snapshots', exist_ok=True)
                filename = f"./snapshots/{base_model}_{likelihood}_{dataset_name}_best.pth.tar"
                print(f"Saving best weights so far with val_loss: {valid_losses[-1]:.5f}")
                torch.save({
                    'epoch': e,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer_net.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': valid_losses,
                }, filename)

            if optimizer_net.param_groups[0]['lr'] < 1e-7:
                break

            save_current_snapshot(base_model, likelihood, dataset_name, e, model, optimizer_net, train_losses, valid_losses, 0, 0)

    except KeyboardInterrupt:
        save_current_snapshot(base_model, likelihood, dataset_name, e-1, model, optimizer_net, train_losses, valid_losses, 0, 0)


if __name__ == '__main__':
    fire.Fire(train)
