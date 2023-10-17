import torch
import torch.nn as nn 


class InceptionBlock(nn.Module):

    def __init__(self, input_channels, hidden_channel_1, hidden_channel_2, out_channel):
        super(InceptionBlock, self).__init__()
        
        # 1 conv 1x1, 2 Conv 3x3
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=hidden_channel_1, 
                kernel_size=1,
                stride=1, 
                padding=0
            ),
            nn.Conv2d(
                in_channels=hidden_channel_1,
                out_channels=hidden_channel_2, 
                kernel_size=3,
                stride=1, 
                padding=1
            ),
            nn.Conv2d(
                in_channels=hidden_channel_2,
                out_channels=out_channel, 
                kernel_size=3,
                stride=1, 
                padding=1
            )
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=hidden_channel_1, 
                kernel_size=1,
                stride=1, 
                padding=0
            ),
            nn.Conv2d(
                in_channels=hidden_channel_1,
                out_channels=out_channel, 
                kernel_size=3,
                stride=1, 
                padding=1
            )
        )
    
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=out_channel, 
                kernel_size=1,
                stride=1, 
                padding=0
            )
        )
        
        self.layer4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=out_channel, 
                kernel_size=1,
                stride=1, 
                padding=0
            )
        )
        
        
    def forward(self, x):
        
        y1 = self.layer1(x)
        y2 = self.layer2(x)
        y3 = self.layer3(x)
        y4 = self.layer4(x)
        x = torch.cat([y1, y2, y3, y4], dim=1)
        return x
    