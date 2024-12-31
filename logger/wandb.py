import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt

from logger import Logger
from datetime import datetime
from logger import TaskType
from typing import Optional


class WandbLogger(Logger):

    def __init__(
        self, 
        task: TaskType, 
        model: nn.Module,
    ):
        wandb.login()
        wandb.init(project="hands-on-monitoring")
        wandb.run.name = f'{task}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

        # Log weights and gradients
        wandb.watch(model, log_freq=100)
        

    def log_classification_training(
        self, 
        epoch: int,
        train_loss_avg:np.ndarray, 
        val_loss_avg: np.ndarray, 
        train_acc_avg: np.ndarray, 
        val_acc_avg: np.ndarray
    ):
        # Log validation metrics
        wandb.log({
            "Classification/val_loss": val_loss_avg,
            "Classification/val_acc": val_acc_avg,
            "epoch": epoch
        })

        # Log training metrics
        wandb.log({
            "Classification/train_loss": train_loss_avg,
            "Classification/train_acc": train_acc_avg,
            "epoch": epoch
        })
        
    def log_confusion_matrix(
        self,
        epoch: int,
        preds: torch.Tensor,
        labels: torch.Tensor
    ):
        pass
        
    def log_model_graph(
        self,
        model: nn.Module,
        batch_sample:torch.Tensor
    ):
        pass
    
    
    def log_embeddings(
        self,
        model:nn.Module,
        data_loader:torch.utils.data.DataLoader,
        func
    ):
        latens,label_img=func(model,data_loader)
        #wandb.log({"embeddings": wandb.Table(dataframe=embeddings)})
        pass 
    
    def log_gradients(
        self,
        epoch:int,
        model:nn.Module
    ):
        pass
        
    def log_image(
        self,
        epoch:int,
        fig: plt.Figure
    ):
        wandb.log({
            "Classification/confusion_matrix": wandb.Image(fig),
            "epoch": epoch
        })