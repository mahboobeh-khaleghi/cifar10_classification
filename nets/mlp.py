import torch # for all things PyTorch
import numpy as np         
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function

class MLP(nn.Module):

    def __init__(self, num_neurons, activations):
        super(MLP, self).__init__()
        
        # constructing layers
        self.layers = self.get_layers(num_neurons, activations)
        
        
    def get_layers(self, num_neurons, activations):
        layers = list()
        for i in np.arange(len(num_neurons)-1) :
            layers.append(
                nn.Linear(num_neurons[i], num_neurons[i+1])
            )
            
            act = self.get_activation(activations[i])
            
            layers.append(act)
            
        return nn.Sequential(*layers)
    
    
    def get_activation(self, act):
        
        act = act.lower()
        
        if act == "relu":
            return nn.ReLU()
        elif act == "leaky_relu":
            return nn.LeakyReLU()
        elif act == "sigmoid":
            return nn.Sigmoid()
        elif act == "tanh":
            return nn.Tanh()
        elif act == "identity":
            return nn.Identity()
        else:
            print(f"Error!! Undefined activation function '{act}'")
            exit()
    
    def forward(self, x):
        # x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        return self.layers(x)