import torch
from train import train
from dataloader import get_dataloaders
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model().to(device)
train_loader, val_loader = get_dataloaders(data_path="data/", split_path="data/")

train(model, train_loader, val_loader, num_epochs=10, device=device)