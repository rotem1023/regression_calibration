
from models import BreastPathQModel, DistancePredictor
import torch

def get_model(model_name, level, base_model, device, after = False):
    models_dir = '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots_new'
    if base_model != None: 
        dist_st = f'dist_{base_model}'
        model = DistancePredictor(model_name).to(device)
        if after:
            dist_st += '_after'
    else:
        dist_st = ''
        model = BreastPathQModel(model_name, out_channels=1).to(device)
    
    checkpoint = torch.load(f'{models_dir}/{model_name}_gaussian_lumbar_L{level}_snapshot_{dist_st}.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model