import math
import pickle
import numpy as np
#%matplotlib inline

import torch
import torchvision.transforms as transforms



class Cifar10DataLoader:
    def __init__(self, pickles_path, transform):
        self.transform = transform
        self.data, self.labels = self.read_batches(pickles_path)
        
    def read_pickle(self, path):
        with open (path, "rb") as file:
            pickle_dict = pickle.load(file , encoding = "latin1")
        return np.array(pickle_dict["labels"]), pickle_dict["data"]
    
    def read_batches(self, pickles_path):
        data = list()
        labels = list()
        
        for path in pickles_path:
            path_label, path_data = self.read_pickle(path)
            data.append(path_data)
            labels.append(path_label)
            
        data = np.concatenate(data).astype(np.float64)
        labels = np.concatenate(labels).astype(np.float64)
        
        data = torch.tensor(data, dtype=torch.float64)
        labels = torch.tensor(labels, dtype=torch.float64)
        
        n_data = data.shape[0]
        n_channel = 3
        wh = data.shape[1] / n_channel
        w = int(math.sqrt(wh))
        h = int(wh / w)
        
        data = data.reshape(data.shape[0], n_channel, h, w)
    
        return data, labels
    
    def get_image_dims(self):
        x = self.data[0, :]
        return x.shape
    
    def get_num_labels(self):
        return self.labels.unique().shape[0]
    
    def __getitem__(self, ix):
        x = self.data[ix,:]
        y = self.labels[ix]
        
        # x = x.reshape(3,32,32)
        y = y.to(dtype=torch.int64)
        
        if self.transform != None:
            x = self.transform(x)    
        
        return x, y
        
    def __len__(self):
        return self.labels.shape[0]
        