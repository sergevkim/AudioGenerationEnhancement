from argparse import ArgumentParser
from pathlib import Path

from agenh.datamodules import YADCDataModule
# from agenh.loggers import NeptuneLogger
from agenh.models import AutoEncoder
from agenh.trainer import Trainer
import os
import sys
import warnings

from config import (
    CommonArguments,
    DataArguments,
    TrainArguments,
    SpecificArguments,
)


def main(args):
    warnings.filterwarnings("ignore")

    print('\n\nDEVICE: {}\n\n'.format(args.device))

    model = AutoEncoder(
        config={'num_features': 65536, 'device': args.device},
    ).to(args.device)

    datamodule = YADCDataModule(
        data_path=None,
        batch_size=64,
        num_workers=args.num_workers,
    )
    datamodule.setup(val_ratio=0.05)
    
    trainer = Trainer(
        logger=None,
        verbose=args.verbose,
        version=args.version,
        max_epoch=1000
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    default_args_dict = {
        **vars(CommonArguments()),
        **vars(DataArguments()),
        **vars(TrainArguments()),
        **vars(SpecificArguments()),
    }

    for arg, value in default_args_dict.items():
        parser.add_argument(
            f'--{arg}',
            type=type(value),
            default=value,
            help=f'<{arg}>, default: {value}',
        )

    args = parser.parse_args()

    main(args)


