import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import matplotlib.pyplot as plt

class DummyLogger:
    
    def log_classification_training(
        self,
        epoch: int,
        train_loss_avg: np.ndarray,
        val_loss_avg: np.ndarray,
        train_acc_avg: np.ndarray,
        val_acc_avg: np.ndarray
    ):
        pass
    
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
        pass