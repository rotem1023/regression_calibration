
from models import BreastPathQModel, BreastPathQModelOneOutput, DistancePredictor, DistancePredictorOneOutput
import torch

def get_model_lumbar(model_name, level, base_model, device, lambda_param =5, loss = "gaussian", one_out = False):
    models_dir = '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots'
    if base_model != None:  # distance model
        models_dir = f"{models_dir}_new"
        if one_out: # model plots one dim output
            model = DistancePredictorOneOutput(model_name).to(device)
            models_dir = f"{models_dir}/one_output"  
        else:
            model = DistancePredictor(model_name).to(device)
        lambda_st = f"_lambda_{lambda_param}"
        checkpoint = torch.load(f'{models_dir}/{model_name}_lumbar_L{level}_snapshot_dist_{base_model}{lambda_st}_new.pth.tar', map_location=device)
    else:
        model = BreastPathQModel(model_name, out_channels=1).to(device) 
        checkpoint = torch.load(f'{models_dir}/{model_name}_{loss}_lumbar_L{level}_best.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def get_model_boneage(model_name, base_model, device, lambda_param =5, loss = 'gaussian', one_out = False):
    models_dir = '/home/dsi/rotemnizhar/dev/regression_calibration/src/models/snapshots'
    if base_model != None:  # distance model
        models_dir = f"{models_dir}_new"
        if one_out: # model plots one dim output
            model = DistancePredictorOneOutput(model_name, in_channels=1).to(device)
            models_dir = f"{models_dir}/one_output"  
        else:
            model = DistancePredictor(model_name, in_channels=1).to(device)
        lambda_st = f"_lambda_{lambda_param}"
        checkpoint = torch.load(f'{models_dir}/{model_name}_boneage_snapshot_dist_{base_model}{lambda_st}_new.pth.tar', map_location=device)
    else:
        if loss == "gaussian":
            model = BreastPathQModel(model_name, in_channels = 1, out_channels=1).to(device) 
        elif loss =="mse":
            model = BreastPathQModelOneOutput(model_name, in_channels = 1, out_channels=1).to(device) 
        else:
            assert False
        checkpoint = torch.load(f'{models_dir}/{model_name}_{loss}_boneage_best.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model