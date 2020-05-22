import torch.nn as nn


class anCoder(torch.nn):
    '''
        Abstract class for defining an encoder and decoder for a VAE architecture
    '''
    @abstractmethod
    def __init__(self, arch, dropout=0.5):
        self.dims = arch
        self.dropout = dropout
        super(anCoder).__init__()
    
    @abstractmethod
    def forward(self, input):
        raise NotImplementedError