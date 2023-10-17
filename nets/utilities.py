import torch.nn as nn 
from .gap import GAP
from .residual_blocks import ResidualBlock1, ResidualBlock2
from .resnext import ResNext
from .inception_blocks import InceptionBlock

def construct_model(model_config):
    model = list()
    
    input_channel = model_config.input_channel
    layers_config = model_config.layers_config
    
    current_channel = input_channel
    
    for layer_config in layers_config:
        
        # extracting layer type
        layer_type = layer_config.split("_")[0]
        layer_type = layer_type.lower()
        
        # layer_type: 'res1_{middle channels}'
        if layer_type == "res1":
            mid_channels = int(layer_config.split("_")[1])
            model.append(
                ResidualBlock1(
                    input_channels = current_channel,
                    mid_channels = mid_channels
                )
            )
        
        # layer_type: 'res2_{middle channels}_{output channel}'
        elif layer_type == "res2":
            mid_channels = int(layer_config.split("_")[1])
            out_channels = int(layer_config.split("_")[2])
            model.append(
                ResidualBlock2(
                    input_channels = current_channel,
                    mid_channels = mid_channels,
                    output_channels = out_channels
                )
            )
            current_channel = out_channels
        
        # layer_type: 'resnext_{number of groups}_{hidden channels}_{group output channels}'
        elif layer_type == "resnext":
            number_of_groups = int(layer_config.split("_")[1])
            hidden_channels = int(layer_config.split("_")[2])
            group_output_channels = int(layer_config.split("_")[3])
            
            model.append(
                ResNext(
                    num_group = number_of_groups,
                    input_channels = current_channel,
                    hidden_channels = hidden_channels,
                    group_output_channels = group_output_channels
                    
                )
            )
            
        #layer_type: 'inception_{hidden_channel_1}_{hidden_channel_2}_{out_channel}'
        elif layer_type == "inception":
            hidden_channels_1 = int(layer_config.split("_")[1])
            hidden_channels_2 = int(layer_config.split("_")[2])
            output_channels = int(layer_config.split("_")[3])
            
            model.append(
                InceptionBlock(
                    input_channels = current_channel,
                    hidden_channel_1 = hidden_channels_1,
                    hidden_channel_2 = hidden_channels_2,
                    out_channel = output_channels
                )
            )
            current_channel = int(output_channels * 4)
            
        # layer_type: 'conv3x3_{output channels}'
        elif layer_type == "conv3x3":
            out_channels = int(layer_config.split("_")[1])
            model.append(
                nn.Conv2d(
                    in_channels = current_channel,
                    out_channels = out_channels,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
                )
            )
            current_channel = out_channels
            
        # layer_type: 'conv1x1_{output channels}'
        elif layer_type == "conv1x1":
            out_channels = int(layer_config.split("_")[1])
            model.append(
                nn.Conv2d(
                    in_channels = current_channel,
                    out_channels = out_channels,
                    kernel_size = 1,
                    stride = 1,
                    padding = 0
                )
            )
            current_channel = out_channels
            
        # layer_type: 'linear_{output neurons}'
        elif layer_type == "linear":
            out_neurons = int(layer_config.split("_")[1])
            model.append(
                nn.Linear(
                    in_features = current_channel,
                    out_features = out_neurons
                )
            )
            current_channel = out_neurons
        
        # layer_type: 'batchnorm'
        elif layer_type == "batchnorm": 
            model.append(
                nn.BatchNorm2d(current_channel)
            )
        
        # layer_type: 'maxpool'
        elif layer_type == "maxpool":
            model.append(
                nn.MaxPool2d(
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
                )
            )
            
        # layer_type: 'avgpool'
        elif layer_type == "avgpool":
            model.append(
                nn.AvgPool2d(
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
                )
            ) 
        
        # layer_type: 'gap'
        elif layer_type == "gap":
            model.append(GAP()) 
                   
        # layer_type: 'relu'
        elif layer_type == "relu":
            model.append(nn.ReLU())
            
        # layer_type: 'leacky_relu"
        elif layer_type == "leaky_relu":
            model.append(nn.LeakyReLU())
            
        # layer_type: 'sigmoid'
        elif layer_type == "sigmoid":
            model.append(nn.Sigmoid())
            
        # layer_type: 'tanh'
        elif layer_type == "tanh":
            model.append(nn.Tanh())
            
        # layer_type: 'identity'
        elif layer_type == "identity":
            model.append(nn.Identity())
            
        else:
            print("Error!!! Undefined layer in the model.")
            exit()
            
    return nn.Sequential(*model)
