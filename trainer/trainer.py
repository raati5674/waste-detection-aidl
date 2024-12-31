import torch
import torch.nn as nn
import numpy as np
from typing import Tuple,Callable
from logger.logger import Logger
from logger.dummyloger import DummyLogger

class Trainer():
    
    def __init__(self,
                 train_loader: torch.utils.data.DataLoader,
                 validation_loader:torch.utils.data.DataLoader,
                 model: nn.Module,
                 optimizer:torch.optim,
                 criterion:nn.functional,
                 logger:Logger=DummyLogger(),
                 func_end_train_epoch:Callable[[int,nn.Module,np.ndarray,np.ndarray,np.ndarray,np.ndarray],None]=None):        
        self.train_loader=train_loader,
        self.validation_loader=validation_loader,        
        self.model=model,
        self.optimizer=optimizer,
        self.criterion=criterion
        self.logger=logger
        self.func_end_train_steep=func_end_train_epoch
        
        if torch.cuda.is_available():
            self.device=device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device=torch.device("cpu")
            
        
        
    def __accuracy__(self,labels, outputs):
        preds = outputs.argmax(-1)
        acc = (preds == labels.view_as(preds)).detach().cpu().numpy().sum().item()
        return acc
    
    def __validation_sample__(self)->Tuple[torch.Tensor,torch.Tensor]:
        inputs,labels=next(iter(self.validation_loader))
        inputs=inputs.to(self.device)
        return self.model(inputs),labels
    
    def __train_single_epoch__(self,epoch: int,log_interval:int) -> Tuple[float, float]:
        pass
    
    @torch.no_grad() # decorator: avoid computing gradients
    def __eval_single_epoch__(self,data_loader:torch.utils.data.DataLoader)->Tuple[float, float]:    
        pass
    
    
    def test(self,test_dataloader:torch.utils.data.DataLoader)->Tuple[float,float]:
        test_loss, test_acc= self.__eval_single_epoch__(test_dataloader)
        return test_loss,test_acc
    
    def __save_graph_model__(self):
        bach,_=next(iter(self.train_loader))
        self.logger.log_model_graph(self.model,bach.to(self.device))        
        
        
    def get_embeddins(self,
                      model:nn.Module,
                      data_loader:torch.utils.data.DataLoader)->Tuple[torch.Tensor,torch.Tensor]:
        return None,None
       
    def train(self,num_epoch:int,log_interval)->Tuple[float,float]:
        train_losses = []
        train_accs = []
        validation_losses = []
        validation_accs = []

        self.__save_graph_model__()

        for epoch in range(1,num_epoch+1):            
            train_loss, train_acc =self.__train_single_epoch__(epoch,log_interval)
            validation_loss, validation_acc =self.__eval_single_epoch__(self.validation_loader)
            
            if self.func_end_train_epoch!=None:
                self.func_end_train_epoch(epoch,self.model,train_loss, train_acc,validation_loss, validation_acc)
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            validation_losses.append(validation_loss)
            validation_accs.append(validation_acc)
            
            self.logger.log_classification_training(epoch,train_loss,validation_loss,train_acc,validation_acc)
            
            preds,labels=self.__validation_sample__()
            self.logger.log_confusion_matrix(epoch=epoch,preds=preds,labels=labels)
            
            self.logger.log_gradients(epoch=epoch,model=self.model)
        
        
        self.logger.log_embeddings(self.model,self.train_loader, self.get_embeddins)
        
        return train_losses,train_accs,validation_losses,validation_accs

