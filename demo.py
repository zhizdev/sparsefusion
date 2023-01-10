import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pytorch3d.renderer import PerspectiveCameras


from sparsefusion.distillation import distillation_loop, get_default_torch_ngp_opt
from utils.camera_utils import RelativeCameraLoader
from utils.common_utils import get_lpips_fn, get_metrics, split_list, normalize, unnormalize
from utils.co3d_dataloader import CO3Dv2Wrapper
from utils.co3d_dataloader import CO3D_ALL_CATEGORIES, CO3D_ALL_TEN
from utils.load_model import load_models
from utils.load_dataset import load_dataset_test
from utils.check_args import check_args

def fit(gpu, args):
    #@ SPAWN DISTRIBUTED NODES
    rank = args.nr * args.gpus + gpu
    print('spawning gpu rank', rank, 'out of', args.gpus)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.cuda.set_device(gpu)
    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(args.exp_dir + '/log/', exist_ok=True)
    os.makedirs(args.exp_dir + '/metrics/', exist_ok=True)
    os.makedirs(args.exp_dir + '/render_imgs/', exist_ok=True)
    os.makedirs(args.exp_dir + '/render_gifs/', exist_ok=True)
    render_imgs_dir = args.exp_dir + '/render_imgs/'
    print('evaluating', args.exp_dir)

    #@ INIT METRICS
    loss_fn_vgg = get_lpips_fn()

    #@ SET CATEGORIES
    if args.category == 'all_ten':
        cat_list = CO3D_ALL_TEN
    elif args.category == 'all':
        cat_list = CO3D_ALL_CATEGORIES
    else:
        cat_list = [args.category]

    #@ LOOP THROUGH CATEGORIES
    for ci, cat in enumerate(cat_list):

        #@ LOAD MODELS
        eft, vae, vldm = load_models(gpu=gpu, args=args, verbose=False)
        use_diffusion = True

        #@ LOAD DATASET
        print(f'gpu {gpu}: setting category to {cat} {ci}/{len(cat_list)}')
        args.category = cat
        test_dataset = load_dataset_test(args, image_size=args.image_size, masked=False)

        #@ SPLIT VAL LIST
        if args.val_list == None:
            args.val_list = torch.arange(len(test_dataset)).long().tolist()

        val_list = split_list(args.val_list, args.gpus)[gpu]
        print(f'gpu {gpu}: assigned idx {val_list}')

        #@ 
        args.val_seed = 0
        context_views = args.context_views

        #@ LOOP THROUGH VAL LIST
        for val_idx in val_list:

            #@ FETCH DATA
            data = test_dataset.__getitem__(val_idx)
            scene_idx = val_idx
            scene_cameras = PerspectiveCameras(R=data['R'],T=data['T'],focal_length=data['f'],principal_point=data['c'],image_size=data['image_size']).cuda(gpu)
            scene_rgb = data['images'].cuda(gpu)
            scene_mask = data['masks'].cuda(gpu)
            scene_valid_region = data['valid_region'].cuda(gpu)

            #@ SET RANDOM INPUT VIEWS
            g_cpu = torch.Generator()
            g_cpu.manual_seed(args.val_seed + val_idx)
            rand_perm = torch.randperm(len(data['R']), generator=g_cpu)
            input_idx = rand_perm[:context_views].long().tolist()
            output_idx = torch.arange(len(data['R'])).long().tolist()
            print('val_idx', val_idx, input_idx)

            #@ CALL DISTILLATION LOOP
            seq_name = f'{cat}_{val_idx:03d}_c{len(input_idx)}'
            opt = get_default_torch_ngp_opt()
            distillation_loop(
                gpu, 
                args,
                opt,
                (eft, vae, vldm),
                args.exp_dir,
                seq_name,
                scene_cameras,
                scene_rgb,
                scene_mask,
                scene_valid_region,
                input_idx,
                use_diffusion=True,
                max_itr=3000,
                loss_fn_vgg=loss_fn_vgg
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-p', '--port', default=1, type=int, metavar='N',
                        help='last digit of port (default: 1234[1])')
    parser.add_argument('-c', '--category', type=str, metavar='S', required=True,
                        help='category')
    parser.add_argument('-r', '--root', type=str, default='data/co3d_toy', metavar='S',
                        help='location of test features')
    parser.add_argument('-d', '--dataset_name', type=str, default='co3d_toy', metavar='S',
                        help='dataset name')
    parser.add_argument('-e', '--eft', type=str, default='-DNE', metavar='S',
                        help='eft ckpt')
    parser.add_argument('-l', '--vldm', type=str, default='-DNE', metavar='S',
                        help='vldm ckpt')
    parser.add_argument('-a', '--vae', type=str, default='-DNE', metavar='S',
                        help='vae ckpt')
    parser.add_argument('-i', '--idx', type=str, default='-DNE', metavar='N',
                        help='evaluataion indicies')
    parser.add_argument('-v', '--input_views', type=int, default=2, metavar='N',
                        help='input views')
    args = parser.parse_args()

    #@ SET MULTIPROCESSING PORTS
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'1234{args.port}'
    print('using port', f'1234{args.port}')

    #@ SET DEFAUL PARAMETERS
    args.use_r = True
    args.encoder = 'resnet18'
    args.num_input = 4
    args.timesteps = 500
    args.objective = 'noise'
    args.scale_factor = 8
    args.image_size = 256
    args.z_scale_factor = 0.18215

    args.server_prefix = 'checkpoints/'
    args.diffusion_exp_name = 'sf'
    args.eft_ckpt = f'{args.server_prefix}/{args.diffusion_exp_name}/{args.category}/ckpt_latest_eft.pt'
    args.vae_ckpt = f'{args.server_prefix}/sd/sd-v1-3-vae.ckpt'
    args.vldm_ckpt = f'{args.server_prefix}/{args.diffusion_exp_name}/{args.category}/ckpt_latest.pt'

    args.context_views = args.input_views
    args.val_list = [0]
    args.exp_dir = 'output/demo/'

    #@ OVERRIDE DEFAULT ARGS WITH INPUTS
    if args.vae != '-DNE':
        args.vae_ckpt = args.vae
    if args.eft != '-DNE':
        args.eft_ckpt = args.eft
    if args.vldm != '-DNE':
        args.vldm_ckpt = args.vldm
    if args.idx != '-DNE':
        try:
            val_list_str = args.idx.split(',')
            args.val_list = []
            for val_str in val_list_str:
                args.val_list.append(int(val_str))
        except:
            print('ERROR: -i --idx arg invalid, please use form 1,2,3')
            print('Exiting...')
            exit(1)

    check_args(args)

    mp.spawn(fit, nprocs=args.gpus, args=(args,))

if __name__ == '__main__':
    main()