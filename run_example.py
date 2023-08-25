import torch
from visualization import save_frames, frames_to_video
from model import CBSModel, GNNModel
import os
import shutil
from data_gen import get_data
from rdkit import Chem

# A few CBS Cats
# '[BH3-][NH+]1[BH-]([O+]=C(C)c2ccccc2)O[C@H]2Cc3ccccc3[C@H]21'
# '[BH3-][N+]12CCC[C@H]1C(c1ccccc1)(c1ccccc1)O[BH-]2[O+]=C(C)c1ccccc1'
# '[BH3-][N+]12CCC[C@H]1C(c1ccc3ccccc3c1)(c1ccc3ccccc3c1)O[BH-]2[O+]=C(C)c1ccccc1'
# '[BH3-][N+]12CCC[C@H]1C(c1cc(C)cc(C)c1)(c1cc(C)cc(C)c1)O[BH-]2[O+]=C(C)c1ccccc1'

# Example molecules
molecules = [Chem.MolFromSmiles('[BH3-][N+]12CCC[C@H]1C(c1ccc3ccccc3c1)(c1ccc3ccccc3c1)O[BH-]2[O+]=C(C)c1ccccc1'),
             Chem.MolFromSmiles('[BH3-][N+]12CCC[C@H]1C(c1cc(C)cc(C)c1)(c1cc(C)cc(C)c1)O[BH-]2[O+]=C(C)c1ccccc1')]
molecule_names = ["CBS-Kat.", "CBS-Kat."]

model_path = 'models/CBS23_08_2023_17_56.pt'
model_mode = 'CBS' # or GNN

if model_mode == 'CBS':
    data_list, _ = get_data(model_mode, molecules)
else:
    data_list = get_data(model_mode, molecules)

# clear images
shutil.rmtree('frames')
os.makedirs('frames')
print(f'Cleared frames')  

if model_mode == 'CBS':
    model = CBSModel(feature_dim=78,
                    hidden_dim=128)
else:
    # Generate an example dataset
    model = GNNModel(feature_dim=78,
                    hidden_dim=128,
                    output_dim=1)

if len(model_path) != 0:
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

# Save individual frames
print('Starting animation ...')
save_frames(data_list, model, molecule_names)
frames_to_video(output_video=f'{model_mode}.mp4')
