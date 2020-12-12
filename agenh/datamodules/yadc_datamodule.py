from pathlib import Path
import os 
from agenh.utils import get_wav_from_abc

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class YADCDataset(Dataset):
    def __init__(self):
        data_path = Path(os.environ.get('YADC_DATASET_PATH') + '/abc')
        files = os.listdir(data_path)
        self.files = []
        bad = ['11797.abc', '160452.abc', '50073.abc']
        for f in files:
            if ('.abc' in f) and (f not in bad):
                self.files.append('{}/{}'.format(data_path, f))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        return get_wav_from_abc(self.files[idx])[:128]

        
class YADCDataModule:
    def __init__(
            self,
            data_path: Path,
            batch_size: int,
            num_workers: int,
        ):
        if data_path is None:
            data_path = Path(os.environ.get('YADC_DATASET_PATH'))
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def prepare_data(
            data_path: Path,
        ):
        pass

    def setup(
            self,
            val_ratio: float,
        ) -> None:
        data = self.prepare_data(
            data_path=self.data_path,
        )
        full_dataset = YADCDataset(
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

