import torch
import torch.nn as nn

class ResNext(nn.Module):
    def __init__(self,num_group,input_channels,hidden_channels, group_output_channels) -> None:
        super().__init__()
        
        # Module properties
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.group_output_channels = group_output_channels
        self.num_group = num_group
        
        # Constructing a group of conv1x1 followed by conv3x3
        self.convs_group = list()
        
        for _ in range(self.num_group):
            self.convs_group.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels = self.input_channels,
                        out_channels = self.hidden_channels,
                        kernel_size = 1,
                        stride = 1,
                        padding = 0
                        
                    ),
                    nn.Conv2d(
                        in_channels = self.hidden_channels,
                        out_channels = self.group_output_channels,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1
                    )  
                )
            )
        
        self.conv_groups = nn.ModuleList(self.convs_group) 
            
        # the last convolution (conv1x1)
        self.output_conv = nn.Conv2d(
            in_channels = self.num_group * self.group_output_channels, 
            out_channels = self.input_channels, 
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
            
        
    
    def forward(self, x):
        
        # Obtaining output of items in the group
        y = list()
        for ix in range(self.num_group):
           y.append(
            self.convs_group[ix](x)
           )
           
           
        # Concating outputs of the group through channels
        z = torch.cat(y, dim=1)
        
        
        # applying output convolution and residual conection
        output = self.output_conv(z) + x
           
        return output  
        