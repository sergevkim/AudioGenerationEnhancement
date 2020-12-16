from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class ProtostarDataset(Dataset):
    def __init__(
            self,
        ):
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(
            self,
            idx: int,
        ):
        pass


class ProtostarDataModule:
    def __init__(
            self,
            data_path: Path,
            batch_size: int,
            num_workers: int,
        ):
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
        full_dataset = ProtostarDataset(
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

