import torch

__all__ = ['leaky_relu1',
           'normal_init',
           'xavier_normal_init',
           'kaiming_normal_init', 'kaiming_uniform_init',
           'nll_criterion_gaussian', 'nll_criterion_laplacian',
           'save_current_snapshot']


def leaky_relu1(x, slope=0.1, a=1):
    x = torch.nn.functional.leaky_relu(x, negative_slope=slope)
    x = -torch.nn.functional.leaky_relu(-x+a, negative_slope=slope)+a
    return x


def normal_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight.data)


def xavier_normal_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)


def kaiming_normal_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data)


def kaiming_uniform_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight.data)


def nll_criterion_gaussian(mu, logvar, target, reduction='mean'):
    loss = torch.exp(-logvar) * torch.pow(target-mu, 2).mean(dim=1, keepdim=True) + logvar
    return loss.mean() if reduction == 'mean' else loss.sum()


def nll_criterion_laplacian(mu, logsigma, target, reduction='mean'):
    loss = torch.exp(-logsigma) * torch.abs(target-mu).mean(dim=1, keepdim=True) + logsigma
    return loss.mean() if reduction == 'mean' else loss.sum()


def save_current_snapshot(base_model, qhigh, dataset, e, model, optimizer_net, train_losses, valid_losses, coverage, avg_length):
    filename = f'/home/dsi/rotemnizhar/dev/regression_calibration/models/{base_model}_{qhigh}_{dataset}_cqr_{e}_new.pth.tar'
    print(f"Saving at epoch: {e}")
    torch.save({
        'epoch': e,
        'state_dict': model.state_dict(),
        'optimizer': optimizer_net.state_dict(),
        'train_losses': train_losses,
        'val_losses': valid_losses,
        'coverage': coverage,
        'avg_len': avg_length
    }, filename)
    
import torch


def avg_len(uncert, q, n_bins=15, outlier=0.0, range=None, single=False):
    device = uncert.device
    
    if single:
        avg_len = (2 * q * uncert).mean()
    else:
        if range == None:
            bin_boundaries = torch.linspace(uncert.min().item(), uncert.max().item(), n_bins + 1, device=device)
        else:
            bin_boundaries = torch.linspace(range[0], range[1], n_bins + 1, device=device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        errors_in_bin_list = []
        avg_uncert_in_bin_list = []
        prop_in_bin_list = []
        uce_per_bin = []

        uce = torch.zeros(1, device=device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |uncertainty - error| in each bin
            in_bin = uncert.gt(bin_lower.item()) * uncert.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()  # |Bm| / n
            prop_in_bin_list.append(prop_in_bin)
            if prop_in_bin.item() > outlier:
                errors_in_bin = errors[in_bin].float().mean()  # err()
                avg_uncert_in_bin = uncert[in_bin].mean()  # uncert()
                uce_in_bin = torch.abs(avg_uncert_in_bin - errors_in_bin) * prop_in_bin
                uce_per_bin.append(uce_in_bin)
                uce += uce_in_bin

                errors_in_bin_list.append(errors_in_bin)
                avg_uncert_in_bin_list.append(avg_uncert_in_bin)

        err_in_bin = torch.tensor(errors_in_bin_list, device=device)
        avg_uncert_in_bin = torch.tensor(avg_uncert_in_bin_list, device=device)
        prop_in_bin = torch.tensor(prop_in_bin_list, device=device)

    if single:
        return avg_len
    else:
        return uce, err_in_bin, avg_uncert_in_bin, prop_in_bin, uce_per_bin