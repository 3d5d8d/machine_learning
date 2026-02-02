# we should inject some noise into mnist. There are the settings for self-made noise injection.
import torch

class AddGaussianNoise(object):
    #Gaussia noise is characterised by N(0, 1) distribution.
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    
    def __call__(self, tensor):
        #y=x+noise
        return tensor + torch.randn_like(tensor)*self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__+"(mean={0}, std={1})".format(self.mean, self.std)

    