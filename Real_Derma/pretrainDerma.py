import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'Main'))
from tools import *
from Agent import *
from CNNnet import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import warnings
import datetime
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    if os.path.split(os.getcwd())[1] != 'Real_Derma':
        os.chdir(os.path.join(os.getcwd(), 'Real_Derma'))

    setseed(0)
    checkDir("Para")
    data = np.load('../Dataset/dermamnist.npz')
    train_images = data['train_images']
    train_labels = data['train_labels']

    trainer = MNISTTrainer(
        batch_size=64,
        learning_rate=0.001,
        num_epochs=100
    )
    trainer.prepare_data(
        train_images=train_images,
        train_labels=train_labels,
    )
    logpath = "pretrain.txt"
    trainer.run_training(isLog=True, path=logpath, log=log, mute=False)
    trainer.save("Para/DermaMNIST.pth")
    trainer.save_("Para/DermaMNIST_rand.pth")
