import torch
from torch import nn
from torchvision import models

class Backbone(nn.Module):
    def __init__(self, backbone_name, num_classes, num_input_channels, 
                 weights=None, weight_handler="copy"):
        """
        A PyTorch module for creating a custom model using an optionally 
        pre-trained torchvision model as a backbone. The model will have 
        modified input and output layers, depending on the parameters given.

        Args:
            backbone_name : str
                The name of the pre-trained torchvision model to use as a 
                backbone.
            num_classes : int
                The number of output classes.
            num_input_channels :int
                The number of input channels.
            weights : str, optional
                The weights to initialize the backbone with. If not given, the 
                backbone will be initialized randomly.
            weight_handler : str, optional
                Specifies how to handle the weights for extra input channels if 
                num_input_channels > 3. Available options are: 'copy': copy 
                weights from the already initialized channels. 'random': 
                initialize weights randomly. Defaults to 'copy'.
        Supported backbones:
            densenet family:
                densenet121, densenet161, densenet169, densenet201
            efficientnet family:
                efficientnet_b0, efficientnet_b1, efficientnet_b2,
                efficientnet_b3, efficientnet_b4, efficientnet_b5,
                efficientnet_b6, efficientnet_b7, 
                efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s,
            resnext family:
                resnext101_32x8d, resnext101_64x4d, resnext50_32x4d  
        """
        super(Backbone, self).__init__()

        assert weight_handler in ["copy", "random"], \
            "Unrecognized 'weight_handler'."

        # Functions to get the first and last layer names based on model type
        def get_first_layer(backbone_name):
            if 'efficientnet' in backbone_name:
                return 'features[0][0]'
            elif 'densenet' in backbone_name:
                return 'features[0]'
            elif 'resnext' in backbone_name:
                return 'conv1'
            else:
                raise ValueError('Unrecognized backbone architecture.')

        def get_last_layer(backbone_name):
            if 'efficientnet' in backbone_name:
                return 'classifier[1]'
            elif 'densenet' in backbone_name:
                return 'classifier'
            elif 'resnext' in backbone_name:
                return 'fc'
            else:
                raise ValueError('Unrecognized model type')

        # Load the backbone model
        self.backbone = getattr(models, backbone_name)(weights=weights)

        # Modify the first convolution layer
        original_conv = eval('self.backbone.' + get_first_layer(backbone_name))
        new_conv = nn.Conv2d(num_input_channels, original_conv.out_channels, 
                             kernel_size=original_conv.kernel_size, 
                             stride=original_conv.stride, 
                             padding=original_conv.padding, 
                             bias=original_conv.bias)
        
        # weight handling
        if weights is not None:
            new_conv.weight.data[:,:3,:,:] = original_conv.weight.data
            if num_input_channels > 3:
                if weight_handler == "copy":
                    new_conv.weight.data[:,3:,:,:] = original_conv.weight.data[
                        :,:num_input_channels-3,:,:
                    ]
                else:
                    nn.init.kaiming_normal_(new_conv.weight.data[:,3:,:,:])

        # Replace the first convolution layer
        exec('self.backbone.' + get_first_layer(backbone_name) + '= new_conv')

        # Modify the classifier layer
        original_fc = eval('self.backbone.' + get_last_layer(backbone_name))
        new_fc = nn.Linear(original_fc.in_features, num_classes)

        # Replace the classifier layer
        exec('self.backbone.' + get_last_layer(backbone_name) + '= new_fc')

    def forward(self, x):
        x = self.backbone(x)
        return x