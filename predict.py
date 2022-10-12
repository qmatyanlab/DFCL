import torch
import torch.nn as nn
import csv

from train import MainTask
from train_contrastive import ContrastiveTask
from data import MainDataModule, ContrastiveDataModule
import matplotlib.pyplot as plt

import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

from pymatgen.core.structure import Molecule
from sklearn.metrics.pairwise import cosine_similarity

# def predict_contra():
model = ContrastiveTask.load_from_checkpoint(
    'lightning_logs/contrastive/resnet_16_layer4_bs32_agb4_lr0.001_temp0.1/version_0/checkpoints/epoch=983-step=31487.ckpt'
)
model.freeze()

encoder = model.encoder
projection = model.projection
fc = model.fc

dm = ContrastiveDataModule(dataset_dir='dataset_10k_65_downsample', batch_size=32, num_workers=0, subset=300,
                           pin_memory=False)
loader = dm.train_dataloader()
d1, d2, s, mol = next(iter(loader))
mols = [Molecule.from_file('XYZ-qm9/{}'.format(f)) for f in mol]
formula = [m.formula for m in mols]

h1 = encoder(d1)
h2 = encoder(d2)

z1 = projection(h1)
z2 = projection(h2)
sp = fc(torch.cat([h1, h2], dim=-1))

h1 = h1.detach().numpy()
h2 = h2.detach().numpy()
z1 = z1.detach().numpy()
z2 = z2.detach().numpy()

tsne = TSNE(n_components=2,
            perplexity=10, learning_rate='auto', init='pca'
            )
tsne_result = tsne.fit_transform(z1)

tsne_result_df = pd.DataFrame({'tSNE_1': tsne_result[:, 0], 'tSNE_2': tsne_result[:, 1],
                               # 'label': formula
                               })
fig, ax = plt.subplots(1)
sns.scatterplot(x='tSNE_1', y='tSNE_2', data=tsne_result_df, ax=ax, s=50,
                # hue='label'
                )
limx = (tsne_result.min(0)[0] - 2, tsne_result.max(0)[0] + 2)
limy = (tsne_result.min(0)[1] - 2, tsne_result.max(0)[1] + 2)
ax.set_xlim(limx)
ax.set_ylim(limy)
ax.set_aspect('equal')
# ax.legend(bbox_to_anchor=(0.95, 1), loc=2, borderaxespad=0.0)
plt.savefig('tsne.svg', dpi=600)

####
# plt.matshow(h1)
# plt.show()
# plt.matshow(h2)
# plt.show()
# plt.matshow(z1)
# plt.show()
# plt.matshow(z2)
# plt.show()

#### cos similarity
# cos = cosine_similarity(z1, z2)
# plt.matshow(cos)
# plt.axis('off')
# plt.title(r'Cosine similarity of z and $\tilde{z}$')
# plt.colorbar(shrink=0.8)
# plt.savefig('cos.svg', dpi=600)
# plt.show()

# dm = MainDataModule(dataset_dir='dataset_10k_65_downsample', batch_size=256, train_ratio=0.8, subset=None, scale=None)
#
# train_loader = dm.train_dataloader()
#
# # val_loader = dm.val_dataloader()
#
# test_loader = dm.test_dataloader()
#
# # model = MainTask.load_from_checkpoint('lightning_logs/energy_scale1/doubleconv_16_layer3_bs128_lr0.001_scale1/version_0/checkpoints/epoch=435-step=13951.ckpt')
# model = MainTask.load_from_checkpoint(
#     # 'lightning_logs/energy_resnet_32_layer4_bs128_lr0.001_subset50000_scratch/version_0/checkpoints/epoch=236-step=37208.ckpt'
#     # 'lightning_logs/energy_resnet_32_layer4_bs128_lr0.001_subset50000/version_1/checkpoints/epoch=530-step=83366.ckpt'
#     # 'lightning_logs/energy_resnet_32_layer4_bs128_lr0.001_subset10000/version_0/checkpoints/epoch=146-step=4703.ckpt'
#     # 'lightning_logs/energy_scale1/resnet_16_layer4_bs128_lr0.001_scale1/version_0/checkpoints/epoch=495-step=15871.ckpt'
#     # 'lightning_logs/energy/resnet_16_layer4_bs128_agb4_lr0.001_tr0.8_scratchFalse_adam_new/version_0/checkpoints/epoch=498-step=17964.ckpt'
#     # 'lightning_logs/energy/resnet_16_layer4_bs128_agb4_lr0.001_tr0.6_scratchFalse_adam_new/version_1/checkpoints/epoch=461-step=12474.ckpt'
#     # 'lightning_logs/energy/resnet_16_layer4_bs128_agb4_lr0.001_tr0.4_scratchFalse_adam_new/version_0/checkpoints/epoch=479-step=8640.ckpt'
#     'lightning_logs/energy/resnet_16_layer4_bs128_agb4_lr0.001_tr0.2_scratchFalse_adam_new/version_0/checkpoints/epoch=467-step=4212.ckpt'
# )
# model.freeze()
# model.to('cuda')
# outs, ts = [], []
# for loader in test_loader:
#     d, t = next(iter(loader))
#     out = model(d.to('cuda'))
#     outs.append(out)
#     ts.append(t)
#
# pred = torch.stack(outs).squeeze(-1).detach().cpu().numpy()
# target = torch.stack(ts).squeeze(-1).detach().cpu().numpy()
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# sns.set_style()
# plt.figure(figsize=(6,6))
# vmin = min(pred.min(), target.min()) - 1
# vmax = max(pred.max(), target.max()) + 1
# sns.scatterplot(pred[0], target[0], label='scale = 1/3')
# sns.scatterplot(pred[1], target[1], label='scale = 1/2')
# sns.scatterplot(pred[2], target[2], label='scale = 1')
# sns.scatterplot(pred[3], target[3], label='scale = 2')
# sns.scatterplot(pred[4], target[4], label='scale = 3')
# plt.plot([vmin, vmax], [vmin,vmax], ls='--', c='grey')
# plt.title('Train ratio: 20%')
# plt.xlabel('Predicted exchange energy (eV)')
# plt.ylabel('Target exchange energy (eV)')
# plt.legend()
# plt.savefig('plot.svg', bbox_inches='tight')
# plt.show()

# ckpt = torch.load('lightning_logs/energy_scale1/resnet_16_layer4_bs128_lr0.001_scale1/version_0/checkpoints/epoch=495-step=15871.ckpt')
# loss = list(ckpt['callbacks'].values())[0]['current_score']
