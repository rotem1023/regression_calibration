import torch
from torch.nn.functional import leaky_relu
from efficientnet_pytorch import EfficientNet
from resnet import resnet50, resnet101
from densenet import densenet121, densenet201
from torch.nn.functional import leaky_relu



class BreastPathQModel(torch.nn.Module):
    def __init__(self, base_model, in_channels=3, out_channels=2, dropout_rate=0.2, pretrained=False):
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

        t_temp = torch.nn.functional.dropout(x, p=self._dropout_p, training=dropout)
        t_temp = leaky_relu(self._fc_mu1(t_temp))
        t_temp = self._fc_mu2(t_temp)
        t_temp_accu = t_temp.unsqueeze(0)
        
        for i in range(T - 1):
            x = self._base_model(input).relu()

            t_temp = torch.nn.functional.dropout(x, p=self._dropout_p, training=dropout)
            t_temp = leaky_relu(self._fc_mu1(t_temp))
            t_temp = self._fc_mu2(t_temp)
            t_temp_accu = torch.cat([t_temp_accu, t_temp.unsqueeze(0)], dim=0)

        t = t_temp_accu.mean(dim=0)

        if test:
            return t_temp_accu.clamp(0, 1)
        else:
            return t