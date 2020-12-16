from pathlib import Path
<<<<<<< HEAD
from typing import Dict, List

import einops
=======
import os 
from agenh.utils import get_wav_from_abc

>>>>>>> 02-generator
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class YADCDataset(Dataset):
<<<<<<< HEAD
    def __init__(
            self,
            wav_paths: List[Path],
            max_length: int = 256,
        ):
        self.max_length = max_length
        self.wav_paths = wav_paths

    def __len__(self) -> int:
        return len(self.wav_paths)

    def __getitem__(self, idx: int) -> Tensor:
        waveform, _ = torchaudio.load(self.wav_paths[idx])
        waveform = waveform[0][:self.max_length] #TODO [0] or .squeeze or nothing?
        waveform = einops.rearrange(
            tensor=waveform,
            pattern='length -> 1 length',
        )

        return waveform

=======
    def __init__(self):
        data_path = Path(os.environ.get('YADC_DATASET_PATH') + '/abc')
        files = os.listdir(data_path)
        self.files = []
        i = 0
        for f in files:
            if '.abc' in f and not '.wav' in f:
                wav_file = get_wav_from_abc(data_path, f)
                self.files.append(wav_file)
            i += 1

            if (i % 100 == 0):
                print('\n\n\n progress: {}/{}\n\n\n'.format(i, len(files)))
                break

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        w, sr = torchaudio.load(self.files[idx])
        return w[0][:128]

        
>>>>>>> 02-generator
class YADCDataModule:
    def __init__(
            self,
            data_path: Path,
<<<<<<< HEAD
            batch_size: int = 1,
            num_workers: int = 1,
        ):
=======
            batch_size: int,
            num_workers: int,
        ):
        if data_path is None:
            data_path = Path(os.environ.get('YADC_DATASET_PATH'))
>>>>>>> 02-generator
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def prepare_data(
            data_path: Path,
<<<<<<< HEAD
        ) -> Dict[str, Path]:
        wav_paths = list(str(p) for p in data_path.glob('*.wav'))

        data = dict(
            wav_paths=wav_paths
        )

        return data
=======
        ):
        pass
>>>>>>> 02-generator

    def setup(
            self,
            val_ratio: float,
        ) -> None:
        data = self.prepare_data(
            data_path=self.data_path,
        )
        full_dataset = YADCDataset(
<<<<<<< HEAD
            wav_paths=data['wav_paths'],
=======
>>>>>>> 02-generator
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

<<<<<<< HEAD

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

=======
>>>>>>> 02-generator
