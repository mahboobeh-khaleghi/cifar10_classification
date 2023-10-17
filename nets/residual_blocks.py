import torch.nn as nn 


class ResidualBlock1(nn.Module):

    def __init__(self, input_channels, mid_channels):
        super(ResidualBlock1, self).__init__()
        
        # 2 Conv 3x3
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=mid_channels, 
                kernel_size=3,
                stride=1, 
                padding=1
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=input_channels, 
                kernel_size=3,
                stride=1, 
                padding=1
            ),
            nn.BatchNorm2d(input_channels)
        )
        
        
    def forward(self, x):
        
        y = self.layer(x)
        x = x + y
        x = nn.functional.relu(x)
        
        return x
    
    
class ResidualBlock2(nn.Module):

    def __init__(self, input_channels, mid_channels, output_channels):
        super(ResidualBlock2, self).__init__()
        
        # 2 Conv 3x3
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=mid_channels, 
                kernel_size=3,
                stride=1, 
                padding=1
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=output_channels, 
                kernel_size=3,
                stride=1, 
                padding=1
            ),
            nn.BatchNorm2d(output_channels)
        )
        
        # Conv 1x1
        self.conv1x1 = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=output_channels, 
            kernel_size=1
        )
        
    def forward(self, x):
        y = self.layer(x)
        x = self.conv1x1(x)
        x = nn.functional.relu(x + y)
        return x