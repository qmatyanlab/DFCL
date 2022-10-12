from argparse import ArgumentParser

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.models.self_supervised.simclr.simclr_module import SyncFunction, Projection
from pl_bolts.optimizers.lars import LARS
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.seed import seed_everything
from torch import Tensor

from data import ContrastiveDataModule
from model import BaseEncoder


class ContrastiveTask(LightningModule):
    def __init__(
            self,
            gpus,
            num_samples,
            batch_size,
            num_nodes=1,
            warmup_epochs=10,
            max_epochs=100,
            temperature=0.1,
            optimizer='adam',
            exclude_bn_bias=False,
            start_lr=0.0,
            learning_rate=1e-3,
            final_lr=0.0,
            weight_decay=1e-6,
            basic_module='doubleconv',
            layer_order='gcr',
            f_maps=(32, 64, 128),
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.basic_module = basic_module
        self.layer_order = layer_order

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

        self.f_maps = f_maps
        self.encoder = BaseEncoder(f_maps=f_maps, basic_module=basic_module, layer_order=layer_order)
        self.projection = Projection(f_maps[-1], f_maps[-1], f_maps[-1] // 2)
        self.fc = nn.Linear(f_maps[-1] * 2, 1)

    def forward(self, batch):
        d1, d2, s, mol = batch

        h1 = self.encoder(d1)
        h2 = self.encoder(d2)

        z1 = self.projection(h1)
        z2 = self.projection(h2)

        contra_loss = self.nt_xent_loss(z1, z2, self.temperature)

        sp = self.fc(torch.cat([h1, h2], dim=-1))
        scale_loss = F.mse_loss(sp, s.view(sp.shape))

        return contra_loss, scale_loss

    def training_step(self, batch, batch_idx):
        contra_loss, scale_loss = self(batch)
        self.log("train_contra_loss", contra_loss, on_step=False, on_epoch=True)
        self.log("train_scale_loss", scale_loss, on_step=False, on_epoch=True)
        return contra_loss + scale_loss

    def validation_step(self, batch, batch_idx):
        contra_loss, scale_loss = self(batch)
        loss = contra_loss + scale_loss
        self.log("val_contra_loss", contra_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_scale_loss", scale_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

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

        if self.optim == "lars":
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError('Optimizer should be adam or lars')

        return optimizer

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model related arguments
        parser.add_argument('--basic_module', default='resnet', type=str)
        parser.add_argument('--layer_order', default='cge', type=str)
        parser.add_argument('--subset', default=None, type=int)
        parser.add_argument('--temperature', default=0.1, type=float)

        # training related arguments
        parser.add_argument('--optimizer', default='adam', type=str)
        parser.add_argument('--max_epochs', '-e', default=1000, type=int)
        parser.add_argument('--max_steps', default=-1, type=int)
        parser.add_argument('--warmup_epochs', default=10, type=int)
        parser.add_argument('--batch_size', '-bs', default=32, type=int)
        parser.add_argument('--exclude_bn_bias', action='store_true')
        parser.add_argument('--fp32', action="store_true")
        parser.add_argument('--weight_decay', default=1e-6, type=float)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--start_lr', default=0, type=float)
        parser.add_argument('--final_lr', default=1e-6, type=float)
        parser.add_argument('--seed', default=12345, type=int)
        parser.add_argument('--accumulate_grad_batches', '-agb', default=8, type=int)

        # device related arguments
        parser.add_argument('--num_nodes', default=1, type=int)
        parser.add_argument('--gpus', default=2, type=int)
        parser.add_argument('--fast_dev_run', default=None, type=int)
        parser.add_argument('--num_workers', default=8, type=int)

        return parser


def main():
    parser = ArgumentParser()
    parser = ContrastiveTask.add_model_specific_args(parser)

    args = parser.parse_args()

    seed_everything(args.seed)

    dm = ContrastiveDataModule(
        dataset_dir='dataset_10k_65_downsample',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset=args.subset
    )

    args.num_samples = dm.num_samples

    model = ContrastiveTask(**args.__dict__)

    # lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=5, monitor="val_loss")
    callbacks = [model_checkpoint]

    logger = TensorBoardLogger('lightning_logs/contrastive',
                               '{}_{}_layer{}_bs{}_agb{}_lr{}_temp{}_translate'.format(
                                   model.encoder.basic_module, model.encoder.f_maps[0], len(model.encoder.f_maps),
                                   args.batch_size, args.accumulate_grad_batches, args.learning_rate, args.temperature))

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
    )

    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
