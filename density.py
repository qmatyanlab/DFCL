import os
from tqdm import tqdm
import pickle
import csv, gzip
import numpy as np
from pyscf import gto, lib, dft
from pyscf.dft import numint
from pyscf.tools.cubegen import Cube
from pymatgen.core.structure import Molecule
from timeout import timeout

ELEMENTS = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12,
            'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23,
            'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33,
            'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
            'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53,
            'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63,
            'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73,
            'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83,
            'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93,
            'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103,
            'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
            'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118}


class Density:
    @timeout(300)
    def __init__(self, mol_file, grid=None, box_len=40.0):
        mol = Molecule.from_file(mol_file)
        mol_center = mol.get_centered_molecule()

        dmax = mol_center.cart_coords.max()
        dmin = mol_center.cart_coords.min()

        if dmax > 6.0 or dmin < -6.0:
            raise ValueError('Too large molecule')

        atoms = [tuple((site.species_string, site.coords)) for site in mol_center]
        self.mol = gto.M(atom=atoms, spin=None, basis='6311g*')
        if grid is None:
            grid = [65] * 3

        elements = [ELEMENTS[e] for e in self.mol.elements]
        if max(elements) > 36:
            raise ValueError('Only support elements less 36 (Kr).')

        if len(self.mol.elements) > 20:
            raise ValueError('Too many atoms.')

        self.grid = grid
        self.spin = self.mol.spin
        self.box_len = box_len

        self._calculate_energy_dm()

    def _calculate_energy_dm(self):
        mf = dft.RKS(self.mol, xc='PBE0,PBE0')  # CCSD (T) , RPA
        self.E_tot = mf.kernel()
        assert mf.converged, 'Calculation not converged'

        mf_c = dft.RKS(self.mol, xc=',PBE0')
        self.E_c = mf_c.kernel()
        assert mf_c.converged, 'Calculation not converged'

        self.E_x = self.E_tot - self.E_c
        assert self.E_x > -200, 'Exchange energy too large'

        dm = mf.make_rdm1()
        if self.spin:
            self.dm = dm.sum(0)
        else:
            self.dm = dm

        self.dm1 = dm


def get_scaled_density(mol, dm, ngrid=65, box_len=40.0, scaling=1.0, translate=None):
    box_len = box_len * scaling
    grid = [ngrid] * 3
    if translate is None:
        translate = np.array([0., 0., 0.])
    cc = Cube(mol, *grid, origin=np.array([-box_len / 2] * 3) + translate, extent=np.array([box_len] * 3))
    coords = cc.get_coords()
    # ngrids = cc.get_ngrids()
    # blksize = min(8000, ngrids)

    # r = np.zeros((ngrids,))
    # for ip0, ip1 in lib.prange(0, ngrids, blksize):
    #     ao = mol.eval_gto('GTOval', coords[ip0:ip1])
    #     r[ip0:ip1] = numint.eval_rho(mol, ao, dm)

    ao = mol.eval_gto('GTOval', coords)
    rho = numint.eval_rho(mol, ao, dm)

    rho = rho.reshape(grid) * scaling ** 3
    return rho


def prepare_raw_data(xyz_dir: str, n_grid: int, output_dir: str, n_data: int = 1000):
    """
    Store density matrices as numpy arrays in pkl format. [mol_name, exchange_energy]
    are stored as pairs in mol_energy.csv
    Parameters
    ----------
    xyz_dir : directory storing .xyz files
    n_grid : number of grids on each direction.
    output_dir : output directory
    n_data : number of molecules
    """
    filenames = os.listdir(xyz_dir)
    mol_energy, mol_names = [], []

    try:
        with open('{}/mol_energy.csv'.format(output_dir), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                mol_energy.append(row)
                mol_names.append(row[0])

    except:
        mol_energy, mol_names = [], []

    for file in tqdm(filenames):

        if file.split('.')[0] in mol_names:
            continue

        if len(mol_energy) >= n_data:
            print('done')
            break

        try:
            den = Density(xyz_dir + '/' + file, grid=[n_grid] * 3)

            molname = file.split('.')[0]

            mol_energy.append([molname, den.E_x])
            with open('{}/{}.pkl'.format(output_dir, molname), 'wb') as f:
                pickle.dump(den.dm1, f)

            with open('{}/mol_energy.csv'.format(output_dir), 'w') as f:
                writer = csv.writer(f)
                for row in mol_energy:
                    writer.writerow(row)

        except:
            continue


def get_density_dataset(rawdata_dir: str, output_dir: str, n_grid: int = 65, box_len: float = 40.0):
    """
    Generate scaled dataset containing density and exchange energy from raw data directoy.
    [mol_name, enedgy] is stored in mol_energy.csv
    Parameters
    ----------
    rawdata_dir : Directory storing raw density matrices data.
    output_dir : Output directory.
    n_grid : number of grids on each direction
    box_len : length of the box around the molecule
    """
    mol_energy, mol_names = [], []
    with open('{}/mol_energy.csv'.format(rawdata_dir), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            mol_energy.append(row)

    existed = os.listdir(output_dir)
    existed_mol = []
    for m in existed:
        mol_scale = m.split('.')[0]
        if mol_scale[-1] == '1':
            existed_mol.append(mol_scale[:-2])

    mol_energy_scaled = []
    for mol_name, energy in tqdm(mol_energy):

        if mol_name in existed_mol:
            continue

        mol = Molecule.from_file('XYZ-qm9/{}.xyz'.format(mol_name))
        mol = mol.get_centered_molecule()
        atoms = [tuple((site.species_string, site.coords)) for site in mol]
        mol = gto.M(atom=atoms, spin=None, basis='6311g*')

        dm = pickle.load(open('{}/{}.pkl'.format(rawdata_dir, mol_name), 'rb'))
        if mol.spin:
            dm = dm.sum(0)

        scales = [1 / 3, 1 / 2, 1, 2, 3]
        scale_strings = ['1|3', '1|2', '1', '2', '3']

        for s, ss in zip(scales, scale_strings):
            d = get_scaled_density(mol, dm, scaling=s, ngrid=n_grid, box_len=box_len)
            d = d.astype('float32')

            with gzip.open('{}/{}_{}.pkl.gz'.format(output_dir, mol_name, ss), 'wb') as f:
                pickle.dump(d, f)

            mol_energy_scaled.append(['_'.join((mol_name, ss)), float(energy) * s])

        with open('{}/mol_energy.csv'.format(output_dir), 'w') as f:
            writer = csv.writer(f)
            for row in mol_energy_scaled:
                writer.writerow(row)


if __name__ == '__main__':
    # prepare_raw_data('XYZ-qm9', 65, 'rawdata', n_data=5000)
    # get_density_dataset('rawdata', 'dataset-10k-129', n_grid=129)

    # import os
    # import gzip
    # import pickle
    # from tqdm import tqdm
    #
    # file_dir = 'dataset-10k'
    #
    # densities = os.listdir(file_dir)
    # for d_file in tqdm(densities):
    #     if not d_file.endswith('.pkl'):
    #         continue
    #
    #     data = pickle.load(open('{}/{}'.format(file_dir, d_file), 'rb'))
    #
    #     with gzip.open('{}/{}.gz'.format(file_dir, d_file), 'wb') as f:
    #         pickle.dump(data.astype('float32'), f)
    #
    #     os.remove('{}/{}'.format(file_dir, d_file))

    dm = pickle.load(open('rawdata/dsgdb9nsd_000034.pkl', 'rb'))
    mol = Molecule.from_file('XYZ-qm9/dsgdb9nsd_000034.xyz')
    mol = mol.get_centered_molecule()
    atoms = [tuple((site.species_string, site.coords)) for site in mol]
    mol = gto.M(atom=atoms, spin=None, basis='6311g*')

    box_len = 40.0
    scaling = 1.0 /3
    ngrid = 65
    translate = np.array([0, 0, 0])
    origin = np.array([-box_len / 2] * 3)

    box_len = box_len * scaling
    grid = [ngrid] * 3
    cc = Cube(mol, *grid, origin=origin + translate, extent=[box_len] * 3)
    coords = cc.get_coords()

    ao = mol.eval_gto('GTOval', coords)
    rho = numint.eval_rho(mol, ao, dm)

    rho = rho.reshape(grid) * scaling ** 3

    import matplotlib.pyplot as plt

    plt.matshow(rho[:, :, 32])
    plt.show()
