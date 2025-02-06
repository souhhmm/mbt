import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='training')
        for video, audio, targets in pbar:
            video = video.to(self.device)
            audio = audio.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(video, audio)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': correct / total
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='validation')
            for video, audio, targets in pbar:
                video = video.to(self.device)
                audio = audio.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(video, audio)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({
                    'loss': running_loss / (pbar.n + 1),
                    'acc': correct / total
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

def train(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)

    for epoch in range(num_epochs):
        print(f"\nepoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = trainer.train_epoch()
        val_loss, val_acc = trainer.validate()
        
        print(f"train loss: {train_loss:.4f} | train acc: {train_acc:.2f}")
        print(f"val loss: {val_loss:.4f} | val acc: {val_acc:.2f}")
