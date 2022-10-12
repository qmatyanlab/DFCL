from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.seed import seed_everything

from data import MainDataModule
from model import BaseEncoder
from train_contrastive import ContrastiveTask


class MainTask(LightningModule):
    def __init__(
            self,
            gpus,
            num_samples,
            batch_size=32,
            num_nodes=1,
            warmup_epochs=10,
            max_epochs=500,
            temperature=0.1,
            optimizer='adam',
            exclude_bn_bias=False,
            start_lr=0.0,
            learning_rate=1e-3,
            final_lr=0.0,
            weight_decay=1e-6,
            freeze_pretrain=False,
            scratch=True,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.temperature = temperature
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.optim = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.freeze_pretrain = freeze_pretrain
        self.scratch = scratch

        if not self.scratch:
            contrastive_task = ContrastiveTask.load_from_checkpoint(
                # 'lightning_logs/contrastive/doubleconv_32_layer3_bs32_agb8_lr0.001_temp0.1/version_0/checkpoints/epoch=998-step=15984.ckpt'
                # 'lightning_logs/contrastive/resnet_16_layer3_bs32_agb8_lr0.001_temp0.1_translate/version_0/checkpoints/epoch=929-step=14880.ckpt'
                # 'lightning_logs/energy_scale1/resnet_16_layer4_bs128_lr0.001_scale1/version_0/checkpoints/epoch=495-step=15871.ckpt'
                # 'lightning_logs/contrastive/resnet_16_layer4_bs32_agb8_lr0.001_temp0.1_translate/version_0/checkpoints/epoch=984-step=15760.ckpt'
                # 'lightning_logs/contrastive/resnet_16_layer4_bs32_agb8_lr0.001_temp0.1_translate/version_1/checkpoints/epoch=985-step=15776.ckpt'
                'lightning_logs/contrastive/doubleconv_32_layer3_bs32_agb8_lr0.001_temp0.1_translate/version_0/checkpoints/epoch=994-step=15920.ckpt'
            )
            if self.freeze_pretrain:
                contrastive_task.freeze()

            self.encoder = contrastive_task.encoder
            self.fc_out = nn.Linear(self.encoder.f_maps[-1], 1)

        else:
            f_maps = (32, 64, 128)
            self.encoder = BaseEncoder(f_maps=f_maps, basic_module='resnet', layer_order='cge')
            self.fc_out = nn.Linear(f_maps[-1], 1)

    def forward(self, d):
        h = self.encoder(d)
        e = self.fc_out(h)
        return e

    def training_step(self, batch, batch_idx):
        data, target = batch
        out = self(data)
        loss = F.mse_loss(out, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        out = self(data)
        loss = F.mse_loss(out, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx):
        data, target = batch
        out = self(data)

        # loss = F.mse_loss(out, target)
        mae = F.l1_loss(out, target)
        return mae

    def test_epoch_end(self, outputs):
        scale_strs = ['1over3', '1over2', '1', '2', '3']
        # losses = torch.stack([x[0][0] for x in outputs])
        maes = torch.stack([x[0] for x in outputs])

        for mae, s in zip(maes, scale_strs):
            # self.log('test_loss_{}'.format(s), loss)
            self.log('test_mae_{}'.format(s), mae)
        # self.log('test_loss', losses.mean())
        self.log('test_mae', maes.mean())

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=("bias", "bn")):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError('Optimizer should be adam or sgd')
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # training related arguments
        parser.add_argument('--optimizer', '-opt', default='adam', type=str)
        parser.add_argument('--max_epochs', '-e', default=500, type=int)
        parser.add_argument('--max_steps', default=-1, type=int)
        parser.add_argument('--warmup_epochs', default=10, type=int)
        parser.add_argument('--batch_size', '-bs', default=32, type=int)
        parser.add_argument('--exclude_bn_bias', action='store_true')
        parser.add_argument('--fp32', action="store_true")
        parser.add_argument('--weight_decay', default=1e-6, type=float)
        parser.add_argument('--learning_rate', '-lr', default=1e-3, type=float)
        parser.add_argument('--start_lr', default=0, type=float)
        parser.add_argument('--final_lr', default=1e-6, type=float)
        parser.add_argument('--seed', default=12345, type=int)
        parser.add_argument('--freeze_pretrain', action='store_true')
        parser.add_argument('--scratch', action='store_true')
        parser.add_argument('--subset', default='50000', type=str)
        parser.add_argument('--train_ratio', '-tr', default=0.8, type=float)
        parser.add_argument('--accumulate_grad_batches', '-agb', default=8, type=int)

        # device related arguments
        parser.add_argument('--num_nodes', default=1, type=int)
        parser.add_argument('--gpus', default=2, type=int)
        parser.add_argument('--fast_dev_run', default=None, type=int)
        parser.add_argument('--num_workers', default=8, type=int)

        return parser


def main():
    parser = ArgumentParser()
    parser = MainTask.add_model_specific_args(parser)

    args = parser.parse_args()

    seed_everything(args.seed)

    dm = MainDataModule(
        seed=args.seed,
        dataset_dir='dataset_10k_65_downsample',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        scale=None,
        subset=int(args.subset) if args.subset != 'all' else None
    )

    args.num_samples = dm.num_samples

    model = MainTask(**args.__dict__)

    # lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=5, monitor="val_loss")
    # callbacks = [model_checkpoint, lr_monitor]
    callbacks = [model_checkpoint]

    logger = TensorBoardLogger('lightning_logs/energy',
                               '{}_{}_layer{}_bs{}_agb{}_lr{}_tr{}_scratch{}_{}_translate'.format(
                                   model.encoder.basic_module,
                                   model.encoder.f_maps[0],
                                   len(model.encoder.f_maps),
                                   args.batch_size,
                                   args.accumulate_grad_batches,
                                   args.learning_rate,
                                   dm.train_ratio,
                                   args.scratch,
                                   args.optimizer))

    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        strategy=DDPStrategy(find_unused_parameters=False) if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        log_every_n_steps=10,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        accumulate_grad_batches=args.accumulate_grad_batches,
        # auto_lr_find=True,
        # auto_scale_batch_size='power',
        # auto_select_gpus=True
    )

    trainer.fit(model, datamodule=dm,
                # ckpt_path='lightning_logs/energy/resnet_32_layer3_bs32_agb8_lr0.001_tr0.8/version_0/checkpoints/epoch=232-step=18406.ckpt'
                )

    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    main()
