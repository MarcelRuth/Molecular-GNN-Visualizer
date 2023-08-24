import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
import torch
from torch_geometric.data import Data

def generate_structure_from_mol(mol):

    # Generate a 3D structure from mol

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    
    # get the first (and in this case, the only) conformer
    conf = mol.GetConformer()
    coordinates = conf.GetPositions()
    coordinates = np.array(coordinates)
    
    return conf, coordinates, mol

def get_bond_pair(mol):
  bonds = mol.GetBonds()
  res = [[],[]]
  for bond in bonds:
    res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
    res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
  return res

def one_of_k_encoding(x, allowable_set):
  if x not in allowable_set:
    raise Exception("input {0} not in allowable set{1}:".format(
        x, allowable_set))
  return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
  """Maps inputs not in the allowable set to the last element."""
  if x not in allowable_set:
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))

def atom_features(atom,
                  xyz = None,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=True,
                  use_partial_charge=False,
                  ):
  if bool_id_feat:
    return np.array([atom_to_id(atom)])
  else:
    from rdkit import Chem
    results = one_of_k_encoding_unk(
      atom.GetSymbol(),
      [
        'C',
        'N',
        'O',
        'S',
        'F',
        'Si',
        'P',
        'Cl',
        'Br',
        'Mg',
        'Na',
        'Ca',
        'Fe',
        'As',
        'Al',
        'I',
        'B',
        'V',
        'K',
        'Tl',
        'Yb',
        'Sb',
        'Sn',
        'Ag',
        'Pd',
        'Co',
        'Se',
        'Ti',
        'Zn',
        'H',  # H?
        'Li',
        'Ge',
        'Cu',
        'Au',
        'Ni',
        'Cd',
        'In',
        'Mn',
        'Zr',
        'Cr',
        'Pt',
        'Hg',
        'Pb',
        'Unknown'
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
      results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                [0, 1, 2, 3, 4])
    if use_chirality:
      try:
        results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
      except:
        results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]
    if use_partial_charge:
      try:
        #print(atom.GetProp('_GasteigerCharge'))
        #print(type(atom.GetProp('_GasteigerCharge')))
        results = results + [float(atom.GetProp('_GasteigerCharge'))]
      except:
        print('Failed to compute GasteigerCharge')

    return np.array(results)
  
def mol2graph(mol,
              xyz_features = False,
              explicit_H=False):
  
  try:
    _, xyz, mol = generate_structure_from_mol(mol)
  except:
     return None
  mol = Chem.RemoveHs(mol)  
  atoms = mol.GetAtoms()

  if xyz_features:
    node_f = [atom_features(atom, pos, explicit_H=explicit_H, use_partial_charge=False) for atom, pos in zip(atoms, xyz) if atom.GetSymbol != 'H']
  else:
    node_f = [atom_features(atom) for atom in atoms]

  xyz = [pos for atom, pos in zip(atoms, xyz) if atom.GetSymbol != 'H']
  atom_list = [atom.GetSymbol() for atom in atoms if atom.GetSymbol != 'H']
  mol = Chem.RemoveHs(mol)

  edge_index = get_bond_pair(mol)

  data = Data(x=torch.tensor(node_f, dtype=torch.float),
              pos=torch.tensor(xyz, dtype=torch.float),
              edge_index=torch.tensor(edge_index, dtype=torch.long),
              )
  return data, atom_list

def small_graph(molecule):
    # Generate a conformer for the molecule if it doesn't have one
    if molecule.GetNumConformers() == 0:
        AllChem.EmbedMolecule(molecule, AllChem.ETKDG())

    # Extract 3D coordinates from the conformer
    conformer = molecule.GetConformer()
    positions = [list(conformer.GetAtomPosition(i)) for i in range(molecule.GetNumAtoms())]
    x = torch.tensor(positions, dtype=torch.float)
    pos = x
    # Convert molecule to adjacency list
    adj_list = Chem.rdmolops.GetAdjacencyMatrix(molecule).tolist()
    edge_index = []
    for i, row in enumerate(adj_list):
        for j, value in enumerate(row):
            if value == 1:
                edge_index.append((i, j))
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, pos=pos)