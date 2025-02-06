import os
import datetime
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        wandb.login(key=os.getenv("WANDB_API_KEY"))

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        if wandb.run is None:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"wabt_{current_time}"

            wandb.init(project="wabt", name=run_name)
            wandb.watch(self.model)
        self.step = 0

        self.best_val_loss = float("inf")
        self.checkpoint_dir = os.path.join("checkpoints", run_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="training")
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

            self.step += 1
            wandb.log(
                {
                    "batch/train_loss": loss.item(),
                    "batch/train_acc": predicted.eq(targets).sum().item()
                    / targets.size(0),
                    "train/running_loss": running_loss / (pbar.n + 1),
                    "train/running_acc": correct / total,
                    "step": self.step,
                }
            )

            pbar.set_postfix(
                {"loss": running_loss / (pbar.n + 1), "acc": correct / total}
            )

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="validation")
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

                self.step += 1
                wandb.log(
                    {
                        "batch/val_loss": loss.item(),
                        "batch/val_acc": predicted.eq(targets).sum().item()
                        / targets.size(0),
                        "val/running_loss": running_loss / (pbar.n + 1),
                        "step": self.step,
                    }
                )

                pbar.set_postfix(
                    {"loss": running_loss / (pbar.n + 1), "acc": correct / total}
                )

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total

        # save checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": epoch_loss,
            "val_acc": epoch_acc,
            "step": self.step,
        }

        # save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(checkpoint, latest_path)

        # save best checkpoint
        if epoch_loss < self.best_val_loss:
            self.best_val_loss = epoch_loss
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(checkpoint, best_path)
            wandb.log({"best_val_loss": self.best_val_loss})

        return epoch_loss, epoch_acc


def train(model, train_loader, val_loader, num_epochs=10, device="cuda"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)

    for epoch in range(num_epochs):
        print(f"\nepoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = trainer.train_epoch()
        val_loss, val_acc = trainer.validate()

        wandb.log(
            {
                "epoch": epoch + 1,
                "epoch/train_loss": train_loss,
                "epoch/train_acc": train_acc,
                "epoch/val_loss": val_loss,
                "epoch/val_acc": val_acc,
            }
        )

        print(f"train loss: {train_loss:.4f} | train acc: {train_acc:.2f}")
        print(f"val loss: {val_loss:.4f} | val acc: {val_acc:.2f}")
