import torch
from torch.nn.functional import leaky_relu
from efficientnet_pytorch import EfficientNet
from resnet import resnet50, resnet101
from densenet import densenet121, densenet201
from utils import leaky_relu1
import torch.nn as nn
import torch.optim as optim


class BreastPathQModel(torch.nn.Module):
    def __init__(self, base_model, in_channels=3, out_channels=1, dropout_rate=0.2, pretrained=False):
        super().__init__()

        assert base_model in ['resnet50', 'resnet101', 'densenet121', 'densenet201', 'efficientnetb0', 'efficientnetb4']

        self._in_channels = in_channels
        self._out_channels = out_channels

        if base_model == 'resnet50':
            if pretrained:
                assert in_channels == 3
                self._base_model = resnet50(pretrained=True, drop_rate=dropout_rate)
            else:
                self._base_model = resnet50(pretrained=False, in_channels=in_channels, drop_rate=dropout_rate)
            fc_in_features = 2048
        if base_model == 'resnet101':
            if pretrained:
                assert in_channels == 3
                self._base_model = resnet101(pretrained=True, drop_rate=dropout_rate)
            else:
                self._base_model = resnet101(pretrained=False, in_channels=in_channels, drop_rate=dropout_rate)
            fc_in_features = 2048
        if base_model == 'densenet121':
            if pretrained:
                assert in_channels == 3
                self._base_model = densenet121(pretrained=True, drop_rate=dropout_rate)
            else:
                self._base_model = densenet121(pretrained=False, drop_rate=dropout_rate, in_channels=in_channels)
            fc_in_features = 1024
        if base_model == 'densenet201':
            if pretrained:
                assert in_channels == 3
                self._base_model = densenet201(pretrained=True, drop_rate=dropout_rate)
            else:
                self._base_model = densenet201(pretrained=False, drop_rate=dropout_rate, in_channels=in_channels)
            fc_in_features = 1920
        if base_model == 'efficientnetb0':
            if pretrained:
                assert in_channels == 3
                self._base_model = EfficientNet.from_pretrained('efficientnet-b0')
            else:
                self._base_model = EfficientNet.from_name('efficientnet-b0', {'in_channels': in_channels})
            fc_in_features = 1280
        if base_model == 'efficientnetb4':
            if pretrained:
                assert in_channels == 3
                self._base_model = EfficientNet.from_pretrained('efficientnet-b4')
            else:
                self._base_model = EfficientNet.from_name('efficientnet-b4', in_channels= in_channels)
            fc_in_features = 1792

        self._fc_mu1 = torch.nn.Linear(fc_in_features, fc_in_features)
        self._fc_mu2 = torch.nn.Linear(fc_in_features, out_channels)
        self._fc_logvar1 = torch.nn.Linear(fc_in_features, fc_in_features)
        # self._fc_logvar2 = torch.nn.Linear(fc_in_features, out_channels)
        self._fc_logvar2 = torch.nn.Linear(fc_in_features, 1)
        # self._fc_logvar2 = torch.nn.Linear(fc_in_features, out_channels=2)

        if 'resnet' in base_model:
            self._base_model.fc = torch.nn.Identity()
        elif 'densenet' in base_model:  # densenet
            self._base_model.classifier = torch.nn.Identity()
        elif 'efficientnet' in base_model:
            self._base_model._fc = torch.nn.Identity()

        self._dropout_T = 25
        self._dropout_p = 0.5

    def forward(self, input, dropout=True, mc_dropout=False, test=False):

        if mc_dropout:
            assert dropout
            T = self._dropout_T
        else:
            T = 1

        x = self._base_model(input).relu()

        mu_temp = torch.nn.functional.dropout(x, p=self._dropout_p, training=dropout)
        mu_temp = leaky_relu(self._fc_mu1(mu_temp))
        mu_temp = self._fc_mu2(mu_temp)
        mu_temp_accu = mu_temp.unsqueeze(0)

        logvar_temp = torch.nn.functional.dropout(x, p=self._dropout_p, training=dropout)
        logvar_temp = leaky_relu(self._fc_logvar1(logvar_temp))
        logvar_temp = self._fc_logvar2(logvar_temp)
        logvar_temp_accu = logvar_temp.unsqueeze(0)
        for i in range(T - 1):
            x = self._base_model(input).relu()

            mu_temp = torch.nn.functional.dropout(x, p=self._dropout_p, training=dropout)
            mu_temp = leaky_relu(self._fc_mu1(mu_temp))
            mu_temp = self._fc_mu2(mu_temp)
            mu_temp_accu = torch.cat([mu_temp_accu, mu_temp.unsqueeze(0)], dim=0)

            logvar_temp = torch.nn.functional.dropout(x, p=self._dropout_p, training=dropout)
            logvar_temp = leaky_relu(self._fc_logvar1(logvar_temp))
            logvar_temp = self._fc_logvar2(logvar_temp)
            logvar_temp_accu = torch.cat([logvar_temp_accu, logvar_temp.unsqueeze(0)], dim=0)

        mu = mu_temp_accu.mean(dim=0)
        muvar = mu_temp_accu.var(dim=0)
        logvar = logvar_temp_accu.mean(dim=0)

        if test:
            return mu_temp_accu.clamp(0, 1), logvar_temp_accu.clamp_max(0), muvar.clamp(0, 1)
        else:
            return mu, logvar, muvar
        
        

class BreastPathQModelOneOutput(nn.Module):
    def __init__(self, base_model, in_channels=3, out_channels=1, dropout_rate=0.2, pretrained=False):
        super().__init__()

        assert base_model in ['resnet50', 'resnet101', 'densenet121', 'densenet201', 'efficientnetb0', 'efficientnetb4']
        self._in_channels = in_channels
        self._out_channels = out_channels

        # Backbone Selection
        if base_model.startswith('resnet'):
            model_fn = resnet50 if base_model == 'resnet50' else resnet101
            self._base_model = model_fn(pretrained=pretrained)
            fc_in_features = 2048

        elif base_model.startswith('densenet'):
            model_fn = densenet121 if base_model == 'densenet121' else densenet201
            self._base_model = model_fn(pretrained=pretrained)
            fc_in_features = 1024 if base_model == 'densenet121' else 1920

        elif base_model.startswith('efficientnet'):
            model_fn = 'efficientnet-b0' if base_model == 'efficientnetb0' else 'efficientnet-b4'
            self._base_model = EfficientNet.from_pretrained(model_fn) if pretrained else EfficientNet.from_name(model_fn)
            fc_in_features = 1280 if base_model == 'efficientnetb0' else 1792

        # Modify first conv layer for different input channels
        if in_channels != 3:
            first_conv = None
            if 'resnet' in base_model:
                first_conv = self._base_model.conv1
            elif 'densenet' in base_model:
                first_conv = self._base_model.features[0]
            elif 'efficientnet' in base_model:
                first_conv = self._base_model._conv_stem
            
            new_conv = nn.Conv2d(in_channels, first_conv.out_channels,
                                 kernel_size=first_conv.kernel_size,
                                 stride=first_conv.stride,
                                 padding=first_conv.padding,
                                 bias=first_conv.bias is not None)
            new_conv.weight = nn.Parameter(first_conv.weight[:, :in_channels].clone())  # Copy weights
            if 'resnet' in base_model:
                self._base_model.conv1 = new_conv
            elif 'densenet' in base_model:
                self._base_model.features[0] = new_conv
            elif 'efficientnet' in base_model:
                self._base_model._conv_stem = new_conv

        # Replace classifier head with Identity
        if 'resnet' in base_model:
            self._base_model.fc = nn.Identity()
        elif 'densenet' in base_model:
            self._base_model.classifier = nn.Identity()
        elif 'efficientnet' in base_model:
            self._base_model._fc = nn.Identity()

        # Regression Head
        self._fc1 = nn.Linear(fc_in_features, fc_in_features // 2)
        self._fc2 = nn.Linear(fc_in_features // 2, out_channels)
        self._dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input, dropout):
        x = self._base_model(input)
        x = torch.relu(x)  # Ensure positive activation
        x = self._dropout(x)
        x = torch.relu(self._fc1(x))
        output = self._fc2(x)
        return output  # Returns a single scalar per input  # Returns a single scalar per input




class DistancePredictor(nn.Module):
    def __init__(self, base_model='resnet50', in_channels=3):
        super(DistancePredictor, self).__init__()
        
        if base_model == 'resnet50':
            self.base_model = resnet50(pretrained=True)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_features, 2)
            nn.init.kaiming_normal_(self.base_model.fc.weight, nonlinearity='relu')
            nn.init.constant_(self.base_model.fc.bias, 0.1)

        elif base_model == 'densenet121':
            self.base_model = densenet121(pretrained=True)
            num_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Linear(num_features, 2)  # Fix: Should output 2 values
            nn.init.kaiming_normal_(self.base_model.classifier.weight, nonlinearity='relu')
            nn.init.constant_(self.base_model.classifier.bias, 0.1)

        else:
            raise NotImplementedError(f"Only resnet50 and densenet121 are supported, got {base_model}")

        # Modify the first convolutional layer if necessary
        if in_channels != 3:
            old_conv = self.base_model.conv1
            self.base_model.conv1 = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, 
                stride=old_conv.stride, 
                padding=old_conv.padding, 
                bias=old_conv.bias is not None
            )

        # Softplus ensures non-negative outputs
        self.activation = nn.Softplus(beta=1)

    def forward(self, x):
        distances = self.base_model(x)  # Predict two distances
        distances = self.activation(distances)  # Ensure non-negativity
        return distances

    
class DistancePredictorOneOutput(nn.Module):
    def __init__(self, base_model='resnet50'):
        super(DistancePredictorOneOutput, self).__init__()
        if base_model == 'resnet50':
            self.base_model = resnet50(pretrained=True)
        elif base_model == 'densenet121':
            self.base_model = densenet121(pretrained=True)
            num_features = self.base_model.classifier.in_features  # For DenseNet, it's 'classifier'
            self.base_model.classifier = nn.Linear(num_features, 1)
        else:
            raise NotImplementedError(f"Only resnet50 is supported, got {base_model}")

        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, 1)
        # nn.init.xavier_uniform_(self.base_model.fc.weight)
        nn. init.kaiming_normal_(self.base_model.fc.weight, nonlinearity='relu')
        nn.init.constant_(self.base_model.fc.bias, 0.1)

        # Add a new layer to predict d+ and d-
        # self.fc = nn.Linear(1, 2)  # Predict two distances
        # self.relu = nn.ReLU()  # Ensure non-negative outputs
        self.relu = nn.Softplus(beta=1)
    
    def forward(self, x):
        distance = self.base_model(x)  # Directly get two outputs (d+ and d-)
        distance = self.relu(distance)  # Apply ReLU to ensure non-negative predictions
        epsilon = 1e-6  # Add a small positive constant for numerical stability
        # distances = distances + epsilon
        return distance