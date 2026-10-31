import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes: int = 7, use_batchnorm: bool = True, dataset_name: str = 'faces', use_dropout: bool = False):
        super().__init__()
        
        first_kernel = 3 if dataset_name == 'faces' else 5
        first_padding = 1 if dataset_name == 'faces' else 2
        
        def build_block(in_channels, out_channels, k_size=3, pad=1, is_last_block=False):
            layers = []
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=pad, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
            if is_last_block:
                layers.append(nn.AdaptiveAvgPool2d(1))
            else:
                layers.append(nn.MaxPool2d(2))
            return layers

        feature_layers = []
        feature_layers.extend(build_block(3, 32, k_size=first_kernel, pad=first_padding))
        feature_layers.extend(build_block(32, 64))
        feature_layers.extend(build_block(64, 128, is_last_block=True))
        
        self.features = nn.Sequential(*feature_layers)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4) if use_dropout else nn.Identity(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)