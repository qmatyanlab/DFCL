import csv
import os

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataset import random_split
from tqdm import tqdm


class DensityContrastiveDataset(Dataset):

    def __init__(self, dataset_dir, seed=42, subset=None):
        self.dataset_dir = dataset_dir
        self.seed = seed
        self.subset = int(subset * 5) if subset is not None else None

        raw_data, self.density_file, self.scale_data = self.__loadfile()

        torch.manual_seed(seed)
        np.random.seed(seed)
        indices = torch.randperm(self.scale_data.shape[0])

        self.density_file = [self.density_file[index] for index in indices]
        self.scale_data = self.scale_data[indices]

    def __len__(self):
        return len(self.density_file)

    def __getitem__(self, index):
        f1, f2 = self.density_file[index]
        d1 = torch.load(os.path.join(self.dataset_dir, f1))
        d2 = torch.load(os.path.join(self.dataset_dir, f2))
        scale = self.scale_data[index]
        mol = '_'.join(f1.split('_')[:2]) + '.xyz'

        _, xidx, yidx, zidx = torch.where(d2 > 0.001)
        xmin, xmax = xidx.aminmax()
        ymin, ymax = yidx.aminmax()
        zmin, zmax = zidx.aminmax()

        xshift = torch.randint(0 - xmin, 65 - xmax, (1,))
        yshift = torch.randint(0 - ymin, 65 - ymax, (1,))
        zshift = torch.randint(0 - zmin, 65 - zmax, (1,))

        d2 = d2.roll((xshift, yshift, zshift), (1, 2, 3))

        return d1, d2, scale, mol

    def __loadfile(self):
        mol_energy_file = os.path.join(self.dataset_dir, 'mol_energy_nwchem.csv')
        assert os.path.exists(mol_energy_file), f'mol_energy file does not exist!'

        n_data = 0
        raw_data = {}
        with open(mol_energy_file, 'r') as f:
            reader = csv.reader(f)
            for mol_scale, energy in reader:

                if self.subset is not None and n_data == self.subset:
                    break

                _, mol_id, scale = mol_scale.split('_')
                if mol_id not in raw_data.keys():
                    raw_data[mol_id] = {scale: energy}
                else:
                    raw_data[mol_id][scale] = energy

                n_data += 1

        choices = np.random.choice(['1|3', '1|2', '1', '2', '3'], size=len(raw_data), replace=True)

        density_file, scale_data = [], []

        for mol_id, scale in tqdm(zip(raw_data.keys(), choices), total=len(choices)):
            if scale == '1|3':
                scale_data.append(1/3)
            elif scale == '1|2':
                scale_data.append(0.5)
            else:
                scale_data.append(float(scale))

            density_file.append(['dsgdb9nsd_{}_1.pth'.format(mol_id), 'dsgdb9nsd_{}_{}.pth'.format(mol_id, scale)])

        scale_data = torch.tensor(scale_data, dtype=torch.float)
        return raw_data, density_file, scale_data


class ContrastiveDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_dir,
            subset=None,
            train_ratio=0.8,
            seed=42,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.dataset_dir = dataset_dir
        self.train_ratio = train_ratio
        self.seed = seed
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.subset = subset
        self.dataset = DensityContrastiveDataset(dataset_dir, seed, subset=subset)
        self._num_samples = int(self.train_ratio * len(self.dataset))

        n_data = len(self.dataset)
        train_split = int(n_data * train_ratio)
        dataset_train, dataset_val = random_split(
            self.dataset,
            [train_split, len(self.dataset) - train_split],
            generator=torch.Generator().manual_seed(seed)
        )

        self._train_dataloader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            generator=torch.Generator().manual_seed(seed)
        )

        self._val_dataloader = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory
        )

    @property
    def num_samples(self):
        return self._num_samples

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader


class DensityDataset(Dataset):

    def __init__(self, dataset_dir, seed=42, subset=None, scale=None):
        self.dataset_dir = dataset_dir
        self.seed = seed
        self.scale = scale
        self.subset = subset

        self.mol_files, self.scale_data, self.target = self.__loadfile()

        torch.manual_seed(seed)
        indices = torch.randperm(self.target.shape[0])

        self.mol_files = [self.mol_files[index] for index in indices]
        self.target = self.target[indices]
        self.scale_data = self.scale_data[indices]

        if scale is None:
            self.scale_indices = torch.stack(
                [(self.scale_data == s).nonzero().flatten() for s in [1 / 3, 1 / 2, 1, 2, 3]])

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        mol_file = self.mol_files[index]
        density_data = torch.load(os.path.join(self.dataset_dir, '{}.pth'.format(mol_file)))
        return density_data, self.target[index]

    def __loadfile(self):
        mol_energy_file = os.path.join(self.dataset_dir, 'mol_energy_nwchem.csv')
        assert os.path.exists(mol_energy_file), f'mol_energy file does not exist!'

        n_data = 0
        mol_scales, energies, scale_data = [], [], []
        with open(mol_energy_file, 'r') as f:
            reader = csv.reader(f)
            for mol_scale, energy in reader:
                if self.subset is not None and n_data >= self.subset:
                    break
                scale = mol_scale.split('_')[-1]

                if self.scale is not None:
                    if scale != self.scale:
                        continue

                if scale == '1|3':
                    scale_data.append(1 / 3)
                elif scale == '1|2':
                    scale_data.append(0.5)
                else:
                    scale_data.append(float(scale))

                mol_scales.append(mol_scale)
                energies.append(float(energy))

                n_data += 1

        energies = torch.tensor(energies, dtype=torch.float).unsqueeze(-1)
        scale_data = torch.tensor(scale_data, dtype=torch.float)
        return mol_scales, scale_data, energies


class MainDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_dir,
            subset=None,
            scale='1',
            train_ratio=0.8,
            test_ratio=0.1,
            seed=42,
            batch_size=32,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.dataset_dir = dataset_dir
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.subset = subset
        self.scale = scale
        self.dataset = DensityDataset(dataset_dir, seed, subset=subset, scale=scale)

        n_data = len(self.dataset)
        train_split = int(n_data * (1 - test_ratio) * train_ratio)
        test_split = int(n_data * test_ratio)
        dataset_train, dataset_val, dataset_test = random_split(
            self.dataset,
            [train_split, len(self.dataset) - train_split - test_split, test_split],
            generator=torch.Generator().manual_seed(seed)
        )

        test_scale_indices = [[i for i in dataset_test.indices if i in scale_index] for scale_index in
                              self.dataset.scale_indices]
        self._num_samples = len(dataset_train)

        train_kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'drop_last': drop_last,
            'pin_memory': pin_memory,
            'generator': torch.Generator().manual_seed(seed)
        }

        val_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'drop_last': drop_last,
            'pin_memory': pin_memory,
        }

        self._train_dataloader = DataLoader(dataset_train, **train_kwargs)
        self._val_dataloader = DataLoader(dataset_val, **val_kwargs)

        dataset_tests = [Subset(self.dataset, test_scale_index) for test_scale_index in test_scale_indices]
        self._test_dataloaders = [DataLoader(d, **val_kwargs) for d in dataset_tests]

    @property
    def num_samples(self):
        return self._num_samples

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloaders


class MainScale1DataModule(LightningDataModule):
    def __init__(
            self,
            dataset_dir,
            subset=None,
            train_ratio=0.8,
            seed=42,
            batch_size=32,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.dataset_dir = dataset_dir
        self.train_ratio = train_ratio
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.subset = subset
        self.dataset = DensityDataset(dataset_dir, seed, subset=subset, scale='1')

        n_data = len(self.dataset)
        train_split = int(n_data * train_ratio)
        dataset_train, dataset_val = random_split(
            self.dataset,
            [train_split, len(self.dataset) - train_split],
            generator=torch.Generator().manual_seed(seed)
        )

        self._num_samples = len(dataset_train)

        train_kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'drop_last': drop_last,
            'pin_memory': pin_memory,
            'generator': torch.Generator().manual_seed(seed)
        }

        val_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'drop_last': drop_last,
            'pin_memory': pin_memory,
        }

        self._train_dataloader = DataLoader(dataset_train, **train_kwargs)
        self._val_dataloader = DataLoader(dataset_val, **val_kwargs)

    @property
    def num_samples(self):
        return self._num_samples

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader
