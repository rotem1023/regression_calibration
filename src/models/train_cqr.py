# Max-Heinrich Laves
# Institute of Mechatronic Systems
# Leibniz Universität Hannover, Germany
# 2019

import fire
import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from tqdm import tqdm
import sys
import os
#from src.data.data_generator_breast import BreastPathQDataset
from data_generator_boneage import BoneAgeDataset
from data_generator_endovis import EndoVisDataset
from data_generator_oct import OCTDataset
from cqr_model import BreastPathQModel
# from models import BreastPathQModel as BreastPathQModelGauss
from utils import kaiming_normal_init
from utils import nll_criterion_gaussian, nll_criterion_laplacian
from utils import save_current_snapshot
from data_generator_lumbar import LumbarDataset

torch.backends.cudnn.benchmark = True


def compute_coverage_len(y_test, y_lower, y_upper):
    """ Compute average coverage and length of prediction intervals

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)

    Returns
    -------

    coverage : float, average coverage
    avg_length : float, average length

    """
    y_test = y_test.unsqueeze(1).mean(dim=1)
    in_the_range = torch.sum((y_test >= y_lower) & (y_test <= y_upper))
    coverage = in_the_range / len(y_test) * 100
    avg_length = torch.mean(abs(y_upper - y_lower))
    return coverage, avg_length


class AllQuantileLoss(nn.Module):
    """ Pinball loss function
    """
    def __init__(self, quantiles):
        """ Initialize

        Parameters
        ----------
        quantiles : pytorch vector of quantile levels, each in the range (0,1)


        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """ Compute the pinball loss

        Parameters
        ----------
        preds : pytorch tensor of estimated labels (n)
        target : pytorch tensor of true labels (n)

        Returns
        -------
        loss : cost function value

        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        
        target = target.unsqueeze(1).mean(dim=1)

        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))

        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


def pinball_loss(pred, target, gamma=0.9):

    loss = 0.0
    for pred_single, target_single in zip(pred, target):
        loss += (target_single - pred_single) * gamma if pred_single < target_single else (pred_single - target_single) * (1.0 - gamma)
    
    return loss


def train(base_model,
          dataset,
          batch_size=32,
          init_lr=0.001,
          epochs=50,
          augment=True,
          valid_size=300,
          lr_patience=20,
          weight_decay=1e-8,
          gpu=3,
          gamma=0.5, 
          level = 2,
          alpha= 0.1):
          
    qlow = alpha/2
    qhigh = 1-alpha/2
    
    use_gauss_model = True
    gauss_pretrained = False
    gauss_freeze = True
    
    print(dataset)

    assert base_model in ['resnet101', 'densenet201', 'efficientnetb4']
    assert dataset in ['breastpathq', 'boneage', 'endovis', 'oct', 'lumbar']
    assert gpu in [0, 1, 2, 3]

    device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
    print("data_set =", dataset)
    print("model =", base_model)
    print("batch_size =", batch_size)
    print("init_lr =", init_lr)
    print("epochs =", epochs)
    print("augment =", augment)
    print("valid_size =", valid_size)
    print("lr_patience =", lr_patience)
    print("weight_decay =", weight_decay)
    print("device =", device)
    print("qhigh =", qhigh)
    print(f"level: {level}")

    writer = SummaryWriter(comment=f"_{dataset}_{base_model}_{gamma}")

    resize_to = (256, 256)

    if dataset == 'breastpathq':
        resize_to = (384, 384)
        in_channels = 3
        out_channels = 1
        pretrained = True

        data_dir = '/media/fastdata/laves/breastpathq/'
        data_set_train = BreastPathQDataset(data_dir=data_dir, augment=augment, resize_to=resize_to, preload=True)
        data_set_valid = BreastPathQDataset(data_dir=data_dir, augment=False, resize_to=resize_to, preload=True)

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
    elif dataset == 'boneage':
        in_channels = 1
        out_channels = 1
        pretrained = False

        # data_dir = '/media/fastdata/laves/rsna-bone-age/'
        # data_dir = '/home/dsi/frenkel2/data/rsna-bone-age'
        data_dir = "C:\lior\studies\master\projects\calibration/regression calibration/rsna-bone-age"
        data_set_train = BoneAgeDataset(data_dir=data_dir, augment=augment, resize_to=resize_to, preload=True)
        data_set_valid = BoneAgeDataset(data_dir=data_dir, augment=False, resize_to=resize_to, preload=False,
                                        preloaded_data=[data_set_train._labels, data_set_train._imgs]
                                        )

        assert len(data_set_train) > 0
        assert len(data_set_valid) > 0

        # indices = torch.randperm(len(data_set_train))
        # train_indices = indices[:len(indices) - 3*valid_size]
        # valid_indices = indices[len(indices) - 3*valid_size:len(indices) - 2*valid_size]
        # test_indices = indices[len(indices) - 2*valid_size:]
        # torch.save(train_indices, f'./{dataset}_train_indices.pth')
        # torch.save(valid_indices, f'./{dataset}_valid_indices.pth')
        # torch.save(test_indices, f'./{dataset}_test_indices.pth')

        # train_indices = torch.load(f'/home/dsi/frenkel2/regression_calibration/data_indices/{dataset}_train_indices.pth')
        # valid_indices = torch.load(f'/home/dsi/frenkel2/regression_calibration/data_indices/{dataset}_valid_indices.pth')
        train_indices = torch.load(f'./data_indices/{dataset}_train_indices.pth')
        valid_indices = torch.load(f'./data_indices/{dataset}_valid_indices.pth')

        train_loader = torch.utils.data.DataLoader(data_set_train, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(train_indices))
        valid_loader = torch.utils.data.DataLoader(data_set_valid, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(valid_indices))
    elif dataset == 'endovis':
        in_channels = 3
        out_channels = 2
        pretrained = True

        # data_dir = '/media/fastdata/laves/EndoVis15_instrument_tracking'
        data_dir = '/home/dsi/frenkel2/data/Tracking_Robotic_Testing/Tracking'

        data_set_train = EndoVisDataset(data_dir=data_dir, mode='train', augment=True, scale=0.5, preload=True)
        data_set_valid = EndoVisDataset(data_dir=data_dir, mode='val', augment=False, scale=0.5, preload=True)

        assert len(data_set_train) > 0
        assert len(data_set_valid) > 0

        print("len(data_set_train)", len(data_set_train))
        print("len(data_set_valid)", len(data_set_valid))

        train_loader = torch.utils.data.DataLoader(data_set_train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(data_set_valid, batch_size=batch_size, shuffle=True)
    elif dataset == 'lumbar':
        in_channels = 3
        out_channels = 2
        pretrained = True

        

        data_set_train = LumbarDataset(level=level, mode='train', augment=True, scale=0.5)
        data_set_valid = LumbarDataset(level=level, mode='valid', augment=False, scale=0.5)

        assert len(data_set_train) > 0
        assert len(data_set_valid) > 0

        print("len(data_set_train)", len(data_set_train))
        print("len(data_set_valid)", len(data_set_valid))

        train_loader = torch.utils.data.DataLoader(data_set_train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(data_set_valid, batch_size=batch_size, shuffle=True)
    elif dataset == 'oct':
        in_channels = 3
        out_channels = 6
        pretrained = True

        data_dir = '/home/dsi/frenkel2/data/3doct-pose-dataset/data'

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

        train_indices = torch.load(f'/home/dsi/frenkel2/regression_calibration/data_indices/{dataset}_train_indices.pth')
        valid_indices = torch.load(f'/home/dsi/frenkel2/regression_calibration/data_indices/{dataset}_valid_indices.pth')

        train_loader = torch.utils.data.DataLoader(data_set_train, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(train_indices))
        valid_loader = torch.utils.data.DataLoader(data_set_valid, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(valid_indices))
    else:
        assert False


    model = BreastPathQModel(base_model, out_channels=2).to(device)
    # models_dir = '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots/cqr'

    # checkpoint = torch.load(f'{models_dir}/{base_model}_lumbar_L{level}_alpha_{alpha}_cqr_new.pth.tar', map_location=device)
    # model.load_state_dict(checkpoint['state_dict'])
    # epochs_finsihed = checkpoint['epoch']
    # epochs = epochs - epochs_finsihed
    # if epochs < 2:
    #     print("epochs < 2")
    #     return
    # print(f"epochs to finsih: {epochs}")
    
    if not pretrained:
        kaiming_normal_init(model)
    
    if dataset == 'breastpathq' or 'boneage':
        optimizer_net = optim.SGD(model.parameters(), lr=init_lr, weight_decay=weight_decay, momentum=0.9)
        print("SGD(model.parameters(), lr=init_lr, weight_decay=weight_decay, momentum=0.9)")
    else:
        optimizer_net = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
        print("AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)")

    print("ReduceLROnPlateau(optimizer_net, patience=lr_patience, factor=0.1)")
    lr_scheduler_net = optim.lr_scheduler.ReduceLROnPlateau(optimizer_net, patience=lr_patience, factor=0.1)
    
    loss_func = AllQuantileLoss([qlow, qhigh])
    # loss_func = nn.MSELoss()

    print("")

    train_losses = []
    valid_losses = []
    batch_counter = 0
    batch_counter_valid = 0
    best_avg_length = 1e10
    best_coverage = 0
    target_coverage = 100.0*(qhigh - qlow)

    try:
        for e in range(epochs):
            model.train()

            epoch_train_loss = []
            t_train = []
            targets_train = []
            is_best = False

            print("lr =", optimizer_net.param_groups[0]['lr'])
            for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
                data, targets = data.to(device), targets.to(device)
                optimizer_net.zero_grad()
                if use_gauss_model:
                    # t_low, t_high, _ = model(data, dropout=True)
                    # t = torch.cat((t_low, t_high), -1)
                    t  = model(data, dropout=True)
                else:
                    t  = model(data, dropout=True)
                loss = loss_func(t, targets).to(device)
                loss.backward()
                epoch_train_loss.append(loss.item())
                optimizer_net.step()

                targets_train.append(targets.detach().cpu())
                t_train.append(t.detach().cpu())

                writer.add_scalar('train/loss', loss.item(), batch_counter)
                batch_counter += 1
                

            epoch_train_loss = np.mean(epoch_train_loss)
            lr_scheduler_net.step(epoch_train_loss)

            targets_train = torch.cat(targets_train, dim=0)
            t_train = torch.cat(t_train, dim=0)

            model.eval()
            epoch_valid_loss = []
            t_valid = []
            targets_valid = []

            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(tqdm(valid_loader)):
                    data, targets = data.to(device), targets.to(device)
                    if use_gauss_model:
                        # t_low, t_high, _ = model(data, dropout=True)
                        # t = torch.cat((t_low, t_high), -1)
                        t  = model(data, dropout=True)
                    else:
                        t  = model(data, dropout=True)
                    loss_valid = loss_func(t, targets).to(device)
                    epoch_valid_loss.append(loss_valid.item())

                    targets_valid.append(targets.detach().cpu())
                    t_valid.append(t.detach().cpu())

                    writer.add_scalar('valid/loss', loss_valid.item(), batch_counter_valid)
                    batch_counter_valid += 1
                    

            epoch_valid_loss = np.mean(epoch_valid_loss)
            targets_valid = torch.cat(targets_valid, dim=0)
            t_valid = torch.cat(t_valid, dim=0)
            
            y_lower = t_valid[:,0]
            y_upper = t_valid[:,1]
            coverage, avg_length = compute_coverage_len(targets_valid, y_lower, y_upper)
            
            if (coverage >= target_coverage) and (avg_length < best_avg_length):
                best_avg_length = avg_length
                best_coverage = coverage
                best_epoch = e
                is_best = True

            print(f"Epoch {e}:")
            print(f"train: loss: {epoch_train_loss:.5f}, pinball: {loss.item():.5f}")
            print(f"valid: loss: {epoch_valid_loss:.5f}, pinball: {loss_valid.item():.5f}")
            print(f"coverage: {coverage:.5f}, avg_len: {avg_length:.5f}")

            # save epoch losses
            train_losses.append(epoch_train_loss)
            valid_losses.append(epoch_valid_loss)

            # if valid_losses[-1] <= np.min(valid_losses):
            #   is_best = True

            if is_best:
                # filename = f"./snapshots/{base_model}_{likelihood}_{dataset}_best.pth.tar"
                filename = f'/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots/cqr/{base_model}_{dataset}_L{level}_alpha_{alpha}_cqr_best.pth.tar'
                print(f"Saving best weights so far with val_loss: {valid_losses[-1]:.5f}")
                torch.save({
                    'epoch': e,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer_net.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': valid_losses,
                    'coverage': coverage,
                    'avg_len': avg_length
                }, filename)

            if optimizer_net.param_groups[0]['lr'] < 1e-7:
                break

            filename = f'/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots/cqr/{base_model}_{dataset}_L{level}_alpha_{alpha}_cqr_new.pth.tar'
            print(f"Saving best weights so far with val_loss: {valid_losses[-1]:.5f}. filename: {filename}")
            torch.save({
                    'epoch': e,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer_net.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': valid_losses,
                    'coverage': coverage,
                    'avg_len': avg_length
                }, filename)
    except KeyboardInterrupt:
        filename = f'/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots/cqr/{base_model}_{dataset}_L{level}_alpha_{alpha}_cqr_new.pth.tar'
        print(f"Saving best weights so far with val_loss: {valid_losses[-1]:.5f}, filename: {filename}")
        torch.save({
                    'epoch': e,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer_net.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': valid_losses,
                    'coverage': coverage,
                    'avg_len': avg_length
                }, filename)

if __name__ == '__main__':
    
    BS=32
    LR=3e-4
    VS=2000
    PT=20
    WD=1e-7

    
    dataset = 'lumbar'
    # efficientnetb4 densenet201
    base_model = 'densenet201'
    level = 3
    epochs=25
    alpha=0.05
    GPU=3
    
    print("Process ID: ", os.getpid())

    
    train(base_model, dataset, batch_size=BS,
          init_lr=LR,
          epochs=epochs,
          augment=True,
          valid_size=VS,
          lr_patience=PT,
          weight_decay=WD,
          gpu=GPU,
          gamma=0.5,
          level = level,
          alpha=alpha)
    print("finished")
    # fire.Fire(train)
