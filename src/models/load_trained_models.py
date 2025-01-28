
from models import BreastPathQModel, DistancePredictor
import torch

def get_model(model_name, level, base_model, device, lambda_param =5, after = False):
    models_dir = '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots'
    if base_model != None: 
        models_dir = f"{models_dir}_new"
        model = DistancePredictor(model_name).to(device)
        if lambda_param ==5:
            lambda_st = ""
        else:
            lambda_st = f"_lambda_{lambda_param}"
        checkpoint = torch.load(f'{models_dir}/{model_name}_lumbar_L{level}_snapshot_dist_{base_model}{lambda_st}_new.pth.tar', map_location=device)
    else:
        model = BreastPathQModel(model_name, out_channels=1).to(device)
        checkpoint = torch.load(f'{models_dir}/{model_name}_gaussian_lumbar_L{level}_best.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model