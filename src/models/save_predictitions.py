
from models import BreastPathQModel
import torch 
from data_generator_lumbar import LumbarDataset
import numpy as np
from tqdm import tqdm


def get_model_path(level, base_model,device):
    models_dir = '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots'
    model = BreastPathQModel(base_model, out_channels=1).to(device)
    checkpoint = torch.load(f'{models_dir}/{base_model}_gaussian_lumbar_L{level}_best.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model



if __name__== '__main__':
    level = 4
    # densenet201, efficientnetb4
    model_name = 'densenet201'
    dataset = 'lumbar'
    group = 'test'
    device = 'cuda:0'
    batch_size = 64
    
    
    model = get_model_path(level, model_name, device)
    data_set_test_original = LumbarDataset(level=level, mode=group, augment=False, scale=0.5)
    # create a data loader
    data_loader_test = torch.utils.data.DataLoader(data_set_test_original, batch_size=batch_size, shuffle=False)
    
    # set the model to evaluation mode
    model.eval()
    # iterate over the data
    all_labels = []
    all_predictions = []
    all_sds = []
    for i, (inputs, labels) in tqdm(enumerate(data_loader_test), total=len(data_loader_test), desc="Running Inference"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward pass
        outputs = model(inputs)
        y = outputs[0]
        sd = outputs[1]
        # store the outputs
        all_predictions.append(y.detach().cpu().numpy())
        all_sds.append(sd.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    # concatenate the results
    all_predictions = np.concatenate(all_predictions)
    all_sds = np.concatenate(all_sds)
    all_labels = np.concatenate(all_labels)
    
    # save the results to a file
    results_dir = '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/results/predictions'
    np.save(f'{results_dir}/{model_name}_{dataset}_{group}_{level}_predictions.npy', all_predictions)
    np.save(f'{results_dir}/{model_name}_{dataset}_{group}_{level}_sds.npy', all_sds)
    np.save(f'{results_dir}/{model_name}_{dataset}_{group}_{level}_labels.npy', all_labels)
        
    