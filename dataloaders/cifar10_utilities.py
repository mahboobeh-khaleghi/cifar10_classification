class Normalizer:
    def __init__(self, min=0, max=255):
        self.min = min
        self.max = max

    def __call__(self, x):
        return (x - self.min) / (self.max - self.min)

class Standardizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, x):
        return (x - self.mean) / self.std