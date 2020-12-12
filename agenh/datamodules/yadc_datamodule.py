from pathlib import Path
from typing import Dict, List

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class YADCDataset(Dataset):
    def __init__(
            self,
            wav_paths: List[Path],
            max_length: int = 128,
        ):
        self.max_length = max_length
        self.wav_paths = wav_paths

    def __len__(self) -> int:
        return len(self.wav_paths)

    def __getitem__(self, idx: int) -> Tensor:
        waveform, _ = torchaudio.load(self.wav_paths[idx])

        return waveform[0][:self.max_length]


class YADCDataModule:
    def __init__(
            self,
            data_path: Path,
            batch_size: int = 1,
            num_workers: int = 1,
        ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def prepare_data(
            data_path: Path,
        ) -> Dict[str, Path]:
        wav_paths = list(str(p) for p in data_path.glob('*.wav'))

        data = dict(
            wav_paths=wav_paths
        )

        return data

    def setup(
            self,
            val_ratio: float,
        ) -> None:
        data = self.prepare_data(
            data_path=self.data_path,
        )
        full_dataset = YADCDataset(
            wav_paths=data['wav_paths'],
        )

        full_size = len(full_dataset)
        val_size = int(val_ratio * full_size)
        train_size = full_size - val_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset=full_dataset,
            lengths=[train_size, val_size],
        )

    def train_dataloader(self) -> DataLoader:
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return val_dataloader

    def test_dataloader(self):
        pass


if __name__ == '__main__':
    datamodule = YADCDataModule(
        data_path=Path('/Users/sergevkim/git/sergevkim/AudioGenerationEnhancement/data/wavs'),
    )
    datamodule.setup(val_ratio=0.1)
    loader = datamodule.train_dataloader()

    for i in loader:
        print(i)
        break
    print(datamodule)

