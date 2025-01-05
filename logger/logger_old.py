import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

class Logger:

    def log_classification_training(
        self,
        epoch: int,
        train_loss_avg: np.ndarray,
        val_loss_avg: np.ndarray,
        train_acc_avg: np.ndarray,
        val_acc_avg: np.ndarray
    ):
        raise NotImplementedError
    
    def log_confusion_matrix(
        self,
        epoch: int,
        preds: torch.Tensor,
        labels: torch.Tensor
    ):
        raise NotImplementedError
      
    def log_model_graph(
        self, 
        model: nn.Module, 
        batch_sample:torch.Tensor
    ):
        raise NotImplementedError

    def log_embeddings(
        self,
        model:nn.Module,
        data_loader:torch.utils.data.DataLoader,
        func
    ):
        raise NotImplementedError
    
    def log_gradients(
        self,
        epoch:int,
        model:nn.Module
    ):
        raise NotImplemented
    
    def log_image(
        self,
        epoch:int,
        fig: plt.Figure
    ):
        raise NotImplemented
    
    def __make_confusion_matrix__(self,preds: torch.Tensor, labels: torch.Tensor) -> plt.Figure:
        predictions = preds.argmax(dim=1, keepdim=True)

        cm = confusion_matrix(labels.cpu(), predictions.cpu())
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='none', cmap=plt.cm.Blues)

        plt.colorbar()
        tick_marks = np.arange(10)

        plt.xticks(tick_marks, np.arange(0, 10))
        plt.yticks(tick_marks, np.arange(0, 10))

        plt.tight_layout()
        threshold = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title("Confusion matrix")
        return fig