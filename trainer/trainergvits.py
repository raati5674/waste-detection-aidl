import torch
import torch.nn as nn
import numpy as np
from typing import Tuple,Callable

from trainer.trainer import Trainer
from logger.logger import Logger
from logger.dummyloger import DummyLogger

class TrainerViTs(Trainer):    
    def __init__(self,
                 train_loader: torch.utils.data.DataLoader,
                 validation_loader:torch.utils.data.DataLoader,
                 model: nn.Module,
                 optimizer:torch.optim,
                 criterion:nn.functional,
                 logger:Logger=DummyLogger(),
                 func_end_train_epoch:Callable[[int,nn.Module,np.ndarray,np.ndarray,np.ndarray,np.ndarray],None]=None):
        super(TrainerViTs,self).__init__(train_loader,validation_loader,model,optimizer,criterion,logger)
        self.train_loader=train_loader
        self.validation_loader=validation_loader
        self.model=model
        self.optimizer=optimizer
        self.criterion=criterion
        self.logger=logger
        self.func_end_train_epoch=func_end_train_epoch
        
        self.model.to(self.device)
        
        
    def __accuracy__(self,labels, outputs):
        preds = outputs.argmax(-1)
        acc = (preds == labels.view_as(preds)).detach().cpu().numpy().sum().item()
        return acc
        
    def __train_single_epoch__(self,epoch: int,log_interval:int) -> Tuple[float, float]:
        self.model.train()
        train_loss = []
        acc = 0.
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
                    
            self.optimizer.zero_grad()
            output =self. model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # Compute metrics
            acc +=self.__accuracy__(outputs=output, labels=target)
            train_loss.append(loss.item())

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader.dataset), loss.item()))
        avg_acc = 100. * acc / len(self.train_loader.dataset)
        print('\nTrain Epoch: {}, Length: {}, Final Accuracy: {:.0f}%'.format(epoch, len(self.train_loader.dataset), avg_acc))

        return np.mean(train_loss), avg_acc

    @torch.no_grad() # decorator: avoid computing gradients
    def __eval_single_epoch__(self,data_loader:torch.utils.data.DataLoader)->Tuple[float, float]:
    
        # Dectivate the train=True flag inside the model
        self.model.eval()

        test_loss = 0
        acc = 0
        for _, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)

            # Apply the loss criterion and accumulate the loss
            test_loss += self.criterion(output, target).item()

            # compute number of correct predictions in the batch
            acc +=self.__accuracy__(outputs=output, labels=target)

        test_loss /= len(data_loader.dataset)
        # Average accuracy across all correct predictions batches now
        test_acc = 100. * acc / len(data_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, acc, len(data_loader.dataset), test_acc,
            ))
        return test_loss, test_acc
    
    def train(self,num_epoch:int,log_interval)->Tuple[float,float]:
        return super().train(num_epoch,log_interval)

    def get_embeddins(self,
                      model:nn.Module,
                      data_loader:torch.utils.data.DataLoader)->Tuple[torch.Tensor,torch.Tensor]:
        data,labels=next(iter(data_loader))
        data,labels=data.to(self.device),labels.to(self.device)
        self.model.eval()
        data=model.compute_embeddins(data)
        return data[0],None