import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from typing import Tuple
from logger.logger import Logger
from torch.utils.tensorboard import SummaryWriter
#import torchvision
from logger.logeractivity import TaskType

class TensorboardLogger(Logger):

    def __init__(self,
                 task: TaskType):
        # Define the folder where we will store all the tensorboard logs
        logdir = os.path.join("logs", f"{task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

        # TODO: Initialize Tensorboard Writer with the previous folder 'logdir'
        self.writer=SummaryWriter(log_dir=logdir)


    def __del__(self):
        print("close writer")
        self.writer.close()
         

    def log_classification_training(
        self, 
        epoch: int,
        train_loss_avg: np.ndarray,
        val_loss_avg: np.ndarray,
        train_acc_avg: np.ndarray,
        val_acc_avg: np.ndarray
    ):
        self.writer.add_scalar('Classification/val_loss', val_loss_avg, epoch)

        self.writer.add_scalar('Classification/val_acc', val_acc_avg, epoch)

        self.writer.add_scalar('Classification/train_loss',train_loss_avg, epoch)

        self.writer.add_scalar('Classification/train_acc',train_acc_avg, epoch)

    def log_confusion_matrix(
        self,
        epoch: int,
        preds: torch.Tensor,
        labels: torch.Tensor
    ):
        fig=self.__make_confusion_matrix__(preds=preds,labels=labels)
        self.writer.add_figure("confusion matrix",fig,global_step=epoch)   


    def log_model_graph(
        self, 
        model: nn.Module, 
        batch_sample:torch.Tensor
    ):
        """
        We are going to log the graph of the model to Tensorboard. For that, we need to
        provide an instance of the model and a batch of images, like you'd
        do in a forward pass.
        """
        predictions=model(batch_sample)
        self.writer.add_graph(model,batch_sample)



    def log_embeddings(
        self,
        model:nn.Module,
        data_loader:torch.utils.data.DataLoader,
        func
    ):
        #hasattr(model, 'get_embeddins') and callable(model.get_embeddins)
        
        latens,label_img=func(model,data_loader) 

        if latens!=None:
            self.writer.add_embedding(latens) if label_img==None else self.writer.add_embedding(latens,label_img=label_img) 

            

    def log_gradients(
        self,
        epoch:int,
        model:nn.Module
    ):
         for name, weight in model.named_parameters():
            self.writer.add_histogram(f"Reconstruction/{name}/value",weight,epoch)
            
            
    def log_image(
        self,
        epoch:int,
        fig: plt.Figure
    ):
        self.writer.add_figure("Classification",figure=fig)