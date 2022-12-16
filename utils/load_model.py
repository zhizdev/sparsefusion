'''
Functions to load models
'''
from collections import OrderedDict
import importlib
from omegaconf import OmegaConf
import torch
from external.imagen_pytorch import Unet
from sparsefusion.eft import EpipolarFeatureTransformer
from sparsefusion.vldm import DDPM

def load_models(gpu, args, verbose=False):
    '''
    Loads models

    Args:
        gpu (int): gpu id
        args (Namespace): model args
        eft_ckpt (str): path to eft checkpoint (optional)
        vae_ckpt (str): path to vae checkpoint (optional)
        vldm_ckpt (str): path to vldm checkpoint (optional)
    Returns:
        eft (PyTorch module): eft feature extractor
        vae (PyTorch module): stable diffusion vae
        vldm (PyTorch module): vldm diffusion modle
    '''
    
    eft = None
    vae = None
    vldm = None

    #! LOAD EFT
    eft = EpipolarFeatureTransformer(use_r=args.use_r, encoder=args.encoder, return_features=True, remove_unused_layers=False).cuda(gpu)
    if args.eft_ckpt is not None:
        checkpoint = torch.load(args.eft_ckpt, map_location='cpu')

        model_dict = eft.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        eft.load_state_dict(model_dict)

        print('LOADING 1/3 loaded eft checkpoint from', args.eft_ckpt)

    else:
        print('LOADING 1/3 initialized eft from scratch')

    
    #! LOAD VAE
    if args.vae_ckpt is not None:
        vae = load_vae(args.vae_ckpt, verbose=verbose).cuda(gpu)
        print('LOADING 2/3 loaded sd vae from', args.vae_ckpt)

    else:
        vae = load_vae(None)
        print('LOADING 2/3 initialized vae from scratch')

    #! LOAD UNet
    channels = 4
    feature_dim = 256
    unet1 = Unet(
        channels=channels,
        dim = 256,
        dim_mults = (1, 2, 4, 4),
        num_resnet_blocks = (2, 2, 2, 2),
        layer_attns = (False, False, False, True),
        layer_cross_attns = (False, False, False, False),
        cond_images_channels = feature_dim,
        attn_pool_text=False
    )

    if verbose:
        total_params = sum(p.numel() for p in unet1.parameters())
        print(f"{unet1.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    
    #! LOAD DIFFUSION
    vldm = DDPM(
        channels=channels,
        unets = (unet1, ),
        conditional_encoder = None,
        conditional_embed_dim = None,
        image_sizes = (32, ),
        timesteps = 500,
        cond_drop_prob = 0.1,
        pred_objectives=args.objective,
        conditional=False,
        auto_normalize_img=False,
        clip_output=True,
        dynamic_thresholding=False,
        dynamic_thresholding_percentile=.68,
        clip_value=10,
    ).cuda(gpu)
    if args.vldm_ckpt is not None:
        checkpoint = torch.load(args.vldm_ckpt, map_location='cpu')
        vldm.load_state_dict(checkpoint['model_state_dict'])
        print('LOADING 3/3 loaded diffusion from', args.vldm_ckpt)
    else:
        print('LOADING 3/3 loaded diffusion from', 'scratch')


    return eft, vae, vldm


def load_vae(vae_ckpt, verbose=False):
    '''
    Load StableDiffusion VAE
    '''
    config = OmegaConf.load(f"external/ldm/configs/sd-vae.yaml")
    vae = load_model_from_config(config, ckpt=vae_ckpt, verbose=verbose)
    vae = vae
    return vae


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model_from_config(config, ckpt=None, verbose=True, full_model=False):

    model = instantiate_from_config(config.model)

    if ckpt is not None:
        if verbose:
            print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            if verbose:
                print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]

        #! Rename state dict params
        sd_ = OrderedDict()
        for k, v in sd.items():
            if not full_model:
                name = k.replace('first_stage_model.','').replace('model.','')
            else:
                name = k
            sd_[name] = v
    
        m, u = model.load_state_dict(sd_, strict=False)

        if len(m) > 0 and verbose:
            print("missing keys:")
            parent_set = set()
            for uk in m:
                uk_ = uk[:uk.find('.')]
                if uk_ not in parent_set:
                    parent_set.add(uk_)
            print(parent_set)
        else:
            if verbose:
                print('all weights found')
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(len(u))
            parent_set = set()
            for uk in u:
                uk_ = uk[:uk.find('.')]
                if uk_ not in parent_set:
                    parent_set.add(uk_)
            print(parent_set)
    else:
        if verbose:
            print('initializing from scratch')

    model.eval()
    return model