import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.FashionMNIST import TraintMNIST,ValidationMNIST
from torch.utils.data import DataLoader

from trainer.trainergvits import TrainerViTs

from model.vits import ViT

from logger.tensorboard import TensorboardLogger
from logger.logeractivity import TaskType
import random
import numpy as np
import os

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

#can cause differents algorithms in subsecuents runs 
# (False reduce performance but use the same algorithm always)
torch.backends.cudnn.benchmark = True


batch_size = 64

train_data=TraintMNIST()
validation_data=ValidationMNIST()

validation_dataset,test_dataset=torch.utils.data.random_split(validation_data,[0.5,0.5])

train_dataloader= DataLoader(train_data, batch_size = batch_size,shuffle=True)
validarion_dataloader = DataLoader(validation_dataset, batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

hyperparameters={
    'img_size': 28, 
    'patch_size': 2, 
    'num_hiddens': 512, 
    'mlp_num_hiddens': 2048, 
    'num_heads': 8,
    'num_blks': 2, 
    'emb_dropout': 0.1, 
    'blk_dropout': 0.1, 
    'lr': 0.1
}


model = ViT(hyperparameters['img_size'],
            hyperparameters['patch_size'],
            hyperparameters['num_hiddens'],
            hyperparameters['mlp_num_hiddens'],
            hyperparameters['num_heads'],
            hyperparameters['num_blks'],
            hyperparameters['emb_dropout'],
            hyperparameters['blk_dropout'])


#optimizer=torch.optim.Adam(model.parameters(),lr=lr)
optimizer=torch.optim.AdamW(model.parameters(),betas=(0.9,0.95),eps=1e-8,lr=0.1)
criterion=nn.CrossEntropyLoss()

logger=TensorboardLogger(TaskType.CLASSIFICATION)

trainer=TrainerViTs(train_dataloader,
                    validarion_dataloader,
                    model,
                    optimizer,
                    criterion,
                    logger=logger)


trainer.train(num_epoch=1,log_interval=100)

print("loss, accuracy in test dataset:\n")
test_loss,test_acc=trainer.test(test_dataloader=test_dataloader)
print(f"loss: {test_loss}, accuracy: {test_acc}")

checkpoint = {
        "model_state_dict":  model.cpu().state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "img_size":hyperparameters['img_size'],
        "patch_size":hyperparameters['patch_size'],
        "num_hiddens":hyperparameters['num_hiddens'],
        "mlp_num_hiddens":hyperparameters['mlp_num_hiddens'],
        "num_heads":hyperparameters['num_heads'],
        "num_blks":hyperparameters['num_blks'],
        "emb_dropout":hyperparameters['emb_dropout'],
        "blk_dropout":hyperparameters['blk_dropout']
}  

if not os.path.exists(f"{os.getcwd()}/app/checkpoint/"):
    os.makedirs(f"{os.getcwd()}/app/checkpoint/")
    
torch.save(checkpoint, f"{os.getcwd()}/app/checkpoint/checkpoint.pt")
