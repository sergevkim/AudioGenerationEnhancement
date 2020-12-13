from argparse import ArgumentParser
from pathlib import Path

from agenh.datamodules import YADCDataModule
from agenh.loggers import NeptuneLogger
from agenh.models import HiFiGANEnhancer
from agenh.trainer import Trainer

from config import (
    CommonArguments,
    DataArguments,
    TrainArguments,
    SpecificArguments,
)


def main(args):
    enhancer = HiFiGANEnhancer(
        learning_rate=args.learning_rate,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_step_size=args.scheduler_step_size,
        verbose=args.verbose,
        device=args.device,
    ).to(args.device)
    datamodule = YADCDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup(val_ratio=args.val_ratio)

    #logger = NeptuneLogger(
    #    api_key=None,
    #    project_name=None,
    #)
    logger = None
    trainer = Trainer(
        logger=logger,
        max_epoch=args.max_epoch,
        verbose=args.verbose,
        version=args.version,
    )

    trainer.fit(
        model=enhancer,
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


