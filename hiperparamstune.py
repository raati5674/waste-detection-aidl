import torch
from torch import nn
from torch.utils.data import DataLoader

import os
import random
import torch
import numpy as np

import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
from ray.train import RunConfig
import tempfile


from datasets.FashionMNIST import TraintMNIST,ValidationMNIST

from model.vits import ViT
from trainer.trainergvits import TrainerViTs
from logger.dummyloger import DummyLogger

#to repeat the experiments.
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
#can cause differents algorithms in subsecuents runs 
# (False reduce performance but use the same algorithm always)
torch.backends.cudnn.benchmark = True

train_data=TraintMNIST()
validation_data=ValidationMNIST()

logger=DummyLogger()

def train_vits(config):
    
    def end_train_epoch(epoch,net,train_loss,validation_loss,train_acc,validation_acc):
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
                        
            if train.get_context().get_world_rank() == 0:
                if (epoch + 1) % 5 == 0:
                    torch.save(
                        {"epoch": epoch,"model_state":net.module.state_dict()},
                        os.path.join(temp_checkpoint_dir, "model.pt"),
                    )
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            train.report({"mean_accuracy": np.mean(validation_acc) }, checkpoint=checkpoint)
                
    batch_size = 64
    
    train_dataloader= DataLoader(train_data, batch_size = batch_size)
    validarion_dataloader = DataLoader(validation_data, batch_size=batch_size)
    
    model = ViT(config["img_size"],
            config["patch_size"],
            config["num_hiddens"],
            config["mlp_num_hiddens"],
            config["num_heads"],
            config["num_blks"],
            config["emb_dropout"],
            config["blk_dropout"])

    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    criterion=nn.CrossEntropyLoss()

    
    trainer=TrainerViTs(train_dataloader,
                        validarion_dataloader,
                        model,
                        optimizer,
                        criterion,
                        logger,
                        func_end_train_epoch=end_train_epoch)

    train_losses,train_accs,validation_losses,validation_accs=trainer.train(2,log_interval=100)
        


search_space = {
    #"lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
    'img_size':28,
    'patch_size':tune.choice([2, 4]),
    'num_hiddens':tune.choice([256, 512, 1024]),
    'mlp_num_hiddens':tune.choice([1024, 2048, 4096]),
    'num_heads':tune.choice([4, 8, 16]),
    'num_blks':tune.choice([1, 2, 4]),
    'emb_dropout':tune.choice([0.1,0.5,0.9]),
    'blk_dropout':tune.choice([0.1,0.5,0.9]),
    'lr':tune.choice([0.01,0.1,1])
}

scheduler = ASHAScheduler(metric="mean_accuracy", mode="max")

#dir logs for tensorboard
logs_dir=f"{os.getcwd()}/logs"

#if no GPU 0 i n GPU n
trainable_with_cpu_gpu = tune.with_resources(train_vits, {"cpu": 1, "gpu": 0})
tune.utils.wait_for_gpu

ray.init(configure_logging=False)

analysis = tune.run(
    trainable_with_cpu_gpu,
    config=search_space,
    storage_path=logs_dir,
    name="tune-hiperparameters",
    scheduler=scheduler,
    num_samples=1
)

print("Best hyperparameters found were: ", analysis.get_best_config(metric="mean_accuracy",mode="max",scope="all"))

ray.shutdown()
