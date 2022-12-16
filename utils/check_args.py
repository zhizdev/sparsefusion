'''
Check that the provided arguments are valid. 
'''

import os
import sys
from utils.co3d_dataloader import CO3D_ALL_CATEGORIES

def check_args(args):

    #@ CHECK ALL EXPECTED ARGS ARE PRESENT
    if not (args.dataset_name == 'co3d' or args.dataset_name == 'co3d_toy'):
        print(f'ERROR: Provided {args.dataset} as dataset, but only (co3d, co3d_toy) are supported.')
        print('Exiting...')
        exit(1)

    if args.category not in CO3D_ALL_CATEGORIES and args.category not in {'all_ten', 'all'}:
        print(f'ERROR: Provided category {args.category} is not in CO3D.')
        print('Exiting...')
        exit(1)

    #@ CHECK DATASET FOLDER EXISTS
    if not os.path.exists(args.root):
        print(f'ERROR: Provided dataset root {args.root} does not exist.')
        print('Exiting...')
        exit(1)

    #@ CHECK MODEL WEIGHTS PATHS EXIST
    if not os.path.exists(args.eft_ckpt):
        print(f'ERROR: Provided EFT weight {args.eft_ckpt} does not exist.')
        print('Exiting...')
        exit(1)

    if not os.path.exists(args.vldm_ckpt):
        print(f'ERROR: Provided VLDM weight {args.vldm_ckpt} does not exist.')
        print('Exiting...')
        exit(1)

    if not os.path.exists(args.vae_ckpt):
        print(f'ERROR: Provided VAE weight {args.vae_ckpt} does not exist.')
        print('Exiting...')
        exit(1)

    return
