from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelTrainer import ModelTrainer
import Resnet as rn

class Net():
    """
    Base network that defines helper functions, summary and mapping to device
    """
	
    def __init__(self, model):
        self.trainer = None
        self.model = model

    def summary(self, input_size): #input_size=(1, 28, 28)
        summary(self.model, input_size=input_size)

    def gotrain(self, optimizer, train_loader, test_loader, dataloader, epochs, statspath, scheduler=None, batch_scheduler=False, L1lambda=0, LossType='CrossEntropyLoss', tb=None):
        self.trainer = ModelTrainer(self.model, optimizer, train_loader, test_loader, dataloader, statspath, scheduler, batch_scheduler, L1lambda, LossType, tb)
        self.trainer.run(epochs)

    def resumerun(self, epochs):
        self.trainer.run(epochs)
        
    def modelload(self, path):
        self.model.load_state_dict(torch.load(path))
        
    def stats(self):
        return self.trainer.stats if self.trainer else None
	  	
    def getmodel(self):
        return self.model if self.model else None
        
    def setmodel(self, model):
        self.model = model