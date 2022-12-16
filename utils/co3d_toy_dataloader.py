'''
Wrapper for a preprocessed and simplified CO3Dv2 dataset
#@ Modified from https://github.com/facebookresearch/pytorch3d
'''

import os
import torch

class CO3Dv2ToyLoader(torch.utils.data.Dataset):

    def __init__(self, root, category):
        super().__init__()

        self.root = root
        self.category = category

        default_path = f'{root}/{category}/{category}_toy.pt'
        if not os.path.exists(default_path):
            print(f'ERROR: toy dataset not found at {default_path}')
            print('Exiting...')
            exit(1)

        dataset = torch.load(default_path)
        self.seq_list = dataset[category]

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):
        return self.seq_list[index]