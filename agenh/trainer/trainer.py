from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.utils as utils
import tqdm
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
            self,
            logger,
            max_epoch: int,
            verbose: bool,
            version: str,
        ):
        self.logger = logger
        self.max_epoch = max_epoch
        self.verbose = verbose
        self.version = version

    def save_checkpoint(
            self,
            model: Module,
            optimizers: Optimizer,
            epoch_idx: int,
            checkpoints_dir: Path,
        ) -> None:
        checkpoint = {
            'model': model,
            'model_state_dict': model.state_dict(),
            'epoch_idx': epoch_idx,
        }

        for i, optimizer in enumerate(optimizers):
            checkpoint[f'optimizer_{i}'] = optimizer
            checkpoint[f'optimizer_{i}_state_dict'] = optimizer.state_dict()

        checkpoint_path = checkpoints_dir / f"v{self.version}-e{epoch_idx}.pt"
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(
            self,
            model: Module,
            optimizers: List[Optimizer],
            checkpoint_path: Path,
        ) -> None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        for i, optimizer in enumerate(optimizers):
            optimizer.load_state_dict(checkpoint[f'optimizer_{i}_state_dict'])

    @torch.enable_grad()
    def training_epoch(
            self,
            model: Module,
            train_dataloader: DataLoader,
            optimizers: List[Optimizer],
            epoch_idx: int,
        ) -> None:
        model.train()
        losses = list()

        for batch_idx, batch in enumerate(tqdm.tqdm(train_dataloader)):
            loss = model.training_step(batch, batch_idx)
            losses.append(loss.item())
            loss.backward()
            utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)

            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()

            model.training_step_end()

        average_loss = sum(losses) / len(losses)

        if self.verbose:
            print(epoch_idx, average_loss)

        model.training_epoch_end(epoch_idx=epoch_idx)

    @torch.no_grad()
    def validation_epoch(
            self,
            model: Module,
            val_dataloader: DataLoader,
            schedulers: List[_LRScheduler],
            epoch_idx: int,
        ) -> None:
        model.eval()
        losses = list()

        for batch_idx, batch in enumerate(tqdm.tqdm(val_dataloader)):
            loss = model.validation_step(batch, batch_idx)
            losses.append(loss.item())
            model.validation_step_end()

        average_loss = sum(losses) / len(losses)

        if self.verbose:
            print(epoch_idx, average_loss)

        for scheduler in schedulers:
            scheduler.step()

        model.validation_epoch_end(epoch_idx=epoch_idx)

    def fit(
            self,
            model: Module,
            datamodule,
        ) -> None:
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        optimizers, schedulers = model.configure_optimizers()

        self.validation_epoch(
            model=model,
            val_dataloader=val_dataloader,
            schedulers=[],
            epoch_idx=0,
        )
        for epoch_idx in range(1, self.max_epoch + 1):
            self.training_epoch(
                model=model,
                train_dataloader=train_dataloader,
                optimizers=optimizers,
                epoch_idx=epoch_idx,
            )
            self.validation_epoch(
                model=model,
                val_dataloader=val_dataloader,
                schedulers=schedulers,
                epoch_idx=epoch_idx,
            )
            if epoch_idx % 5 == 0:
                self.save_checkpoint(
                    model=model,
                    optimizers=optimizers,
                    epoch_idx=epoch_idx,
                    checkpoints_dir=Path.cwd() / "models",
                )

    @torch.no_grad()
    def predict(
            self,
            model: Module,
            datamodule,
        ) -> List[Tensor]:
        test_dataloader = datamodule.test_dataloader()

        predicts = list()

        for batch_idx, batch in enumerate(test_dataloader):
            predict = model.test_step(batch, batch_idx)
            predicts.append(predict)
            model.test_step_end()

        model.test_epoch_end()

        return predicts


if __name__ == '__main__':
    trainer = Trainer(
        logger=None,
        max_epoch=1,
        verbose=True,
        version='0',
    )
    print(trainer)

