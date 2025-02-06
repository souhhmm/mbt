import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torchaudio
from pathlib import Path
import torchvision.io as io


class UCF101Dataset(Dataset):
    def __init__(self, data_path, split_path, split="train", num_frames=8, t=4):
        self.data_path = Path(data_path)
        self.num_frames = num_frames
        self.t = t
        # read split file
        split_file = "trainlist01.txt" if split == "train" else "testlist01.txt"
        with open(os.path.join(split_path, split_file), "r") as f:
            self.video_list = [line.strip().split(" ")[0] for line in f.readlines()]

        self.video_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # create class to index mapping
        self.class_to_idx = {}
        classes = sorted(
            list(set(video_name.split("/")[0] for video_name in self.video_list))
        )
        for idx, classname in enumerate(classes):
            self.class_to_idx[classname] = idx

    def _load_video(self, video_path):
        try:
            vframes, _, _ = io.read_video(str(video_path), pts_unit="sec")
            total_frames = len(vframes)

            # ensure we don't sample beyond video length
            if total_frames < self.num_frames:
                indices = torch.linspace(0, total_frames - 1, total_frames).long()
                indices = torch.cat(
                    [
                        indices,
                        torch.tensor(
                            [total_frames - 1] * (self.num_frames - total_frames)
                        ),
                    ]
                )
            else:
                indices = torch.linspace(0, total_frames - 1, self.num_frames).long()

            frames = []
            for idx in indices:
                frame = vframes[idx]
                frame = Image.fromarray(frame.numpy())
                frame = self.video_transform(frame)
                frames.append(frame)

        except Exception as e:
            print(e)
            frames = [torch.zeros(3, 224, 224) for _ in range(self.num_frames)]

        return torch.stack(frames)  # [num_frames, c, h, w]

    def _load_audio(self, video_path):
        try:
            audio_array, sample_rate = torchaudio.load(str(video_path))
        except (RuntimeError, TypeError):
            # create a small amount of noise instead of pure zeros
            audio_array = torch.randn(1, 16000 * self.t) * 1e-4
            sample_rate = 16000

        # convert to mono
        if audio_array.shape[0] > 1:
            audio_array = torch.mean(audio_array, dim=0, keepdim=True)

        # resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_array = resampler(audio_array)

        target_length = self.t * 16000
        if audio_array.shape[1] < target_length:
            # pad with zeros if audio is too short
            audio_array = torch.nn.functional.pad(
                audio_array, (0, target_length - audio_array.shape[1])
            )
        else:
            # trim if audio is too long
            audio_array = audio_array[:, :target_length]

        # create mel spectrogram
        # win_length = 0.025 * 16000 = 400 samples (25ms)
        # hop_length = 0.010 * 16000 = 160 samples (10ms)
        spectogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=128,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
        )(audio_array)

        spectogram = torchaudio.transforms.AmplitudeToDB()(spectogram)
        spectogram = spectogram.squeeze(0)  # remove channel dimension

        if spectogram.shape[1] > 400:
            spectogram = spectogram[:, :400]
        elif spectogram.shape[1] < 400:
            spectogram = torch.nn.functional.pad(
                spectogram, (0, 400 - spectogram.shape[1])
            )

        # mean=0 std=0.5 according to ast
        spectogram = (spectogram - spectogram.mean()) / (spectogram.std() + 1e-6) * 0.5

        return spectogram.unsqueeze(0)  # add channel dimension back [1, 128, 100*t]

    def __getitem__(self, idx):
        video_name = self.video_list[idx]
        video_path = self.data_path / "UCF-101" / video_name

        label = video_name.split("/")[0]
        video_tensor = self._load_video(video_path)
        audio_tensor = self._load_audio(video_path)
        class_idx = self.class_to_idx[label]

        return video_tensor, audio_tensor, class_idx

    def __len__(self):
        return len(self.video_list)


def get_dataloaders(data_path, split_path, batch_size=2, num_workers=0):

    train_dataset = UCF101Dataset(
        data_path=data_path,
        split_path=os.path.join(split_path, "ucfTrainTestlist"),
        split="train",
    )

    val_dataset = UCF101Dataset(
        data_path=data_path,
        split_path=os.path.join(split_path, "ucfTrainTestlist"),
        split="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
