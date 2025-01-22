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
                # self._base_model = timm.create_model('efficientnet_b4', pretrained=True)
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
        


class DistancePredictor(nn.Module):
    def __init__(self, base_model='resnet50'):
        super(DistancePredictor, self).__init__()
        if base_model == 'resnet50':
            self.base_model = resnet50(pretrained=True)
        elif base_model == 'densenet121':
            self.base_model = densenet121(pretrained=True)
            num_features = self.base_model.classifier.in_features  # For DenseNet, it's 'classifier'
            self.base_model.classifier = nn.Linear(num_features, 1)
        else:
            raise NotImplementedError(f"Only resnet50 is supported, got {base_model}")

        # Modify the final layer of resnet50 to output a single value (y_hat)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, 2)
        nn.init.xavier_uniform_(self.base_model.fc.weight)
        nn.init.constant_(self.base_model.fc.bias, 0.1)

        # Add a new layer to predict d+ and d-
        # self.fc = nn.Linear(1, 2)  # Predict two distances
        # self.relu = nn.ReLU()  # Ensure non-negative outputs
        self.relu = nn.Softplus(beta=1)
    
    def forward(self, x):
        distances = self.base_model(x)  # Directly get two outputs (d+ and d-)
        distances = self.relu(distances)  # Apply ReLU to ensure non-negative predictions
        epsilon = 1e-6  # Add a small positive constant for numerical stability
        distances = distances + epsilon
        return distances