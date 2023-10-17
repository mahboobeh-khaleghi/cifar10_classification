import torch.nn as nn

from .utilities import construct_model            


class MyModel(nn.Module):

    def __init__(self, model_config, vectorize_input = False):
        super(MyModel, self).__init__()
        
        self.model_config = model_config
        self.vectorize_input = vectorize_input
        
        # constructing layers
        self.model = construct_model(model_config = model_config)
    
    def forward(self, x):
        if self.vectorize_input:
            x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        
        return self.model(x)