'''
Early Access Training Code
'''
import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from einops import rearrange, reduce, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from pytorch3d.renderer import (
    MonteCarloRaysampler,
    GridRaysampler,
    PerspectiveCameras
)

from utils.camera_utils import RelativeCameraLoader
from utils.common_utils import huber, sample_images_at_mc_locs
from utils.render_utils import CustomImplicitRenderer
from utils.render_utils import LightFieldRaymarcher
from utils.load_model import load_vae
from utils.co3d_dataloader import CO3Dv2Wrapper

from external.imagen_pytorch import Unet
from sparsefusion.eft import EpipolarFeatureTransformer
from sparsefusion.vldm import DDPM

def save_visualization(
    eft, diffusion, vae, scale_factor, camera,
    target_image,
    loss_history_color,
    renderer_grid,
    dir='output/test',
    mod='',
    input_rgb=None,
    input_cameras=None,
    z=None,
    args=None
    ):
    """
    Save visualization
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    with torch.no_grad():

        if args.train_eft:
            epipolar_features , _, _ = renderer_grid(
                cameras=camera, 
                volumetric_function=eft.module.batched_forward
            )
        else:
            epipolar_features , _, _ = renderer_grid(
                cameras=camera, 
                volumetric_function=eft.batched_forward
            )
 
        rendered_image, epipolar_latents = (
            epipolar_features.split([3, 256], dim=-1)
        )

        epipolar_latents = rearrange(epipolar_latents, 'b h w f -> b f h w')
        rand = torch.randn_like(z)
        diffusion_z = diffusion.module.sample(cond_images=epipolar_latents, batch_size=1)
        diffusion_z = 1.0 / args.z_scale_factor * diffusion_z
        diffusion_image = unnormalize(vae.decode(diffusion_z)).clip(0.0, 1.0)
        diffusion_image = rearrange(diffusion_image, 'b c h w -> b h w c')

    num_input = len(input_rgb)
    # Generate plots.
    fig, ax = plt.subplots(1, num_input+3, figsize=(2*(num_input+3), 3))
    ax = ax.ravel()
    clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
    for ai in range(num_input):
        ax[ai].imshow(clamp_and_detach(input_rgb[ai]))
        ax[ai].grid("off")
        ax[ai].axis("off")
        ax[ai].set_title('input')

    ax[-3].imshow(clamp_and_detach(target_image[0]))
    ax[-3].grid("off")
    ax[-3].axis("off")
    ax[-3].set_title('target')
    ax[-2].imshow(clamp_and_detach(rendered_image[0]))
    ax[-2].grid("off")
    ax[-2].axis("off")
    ax[-2].set_title('pred')
    ax[-1].imshow(clamp_and_detach(diffusion_image[0]))
    ax[-1].grid("off")
    ax[-1].axis("off")
    ax[-1].set_title('pred')

    plt.savefig(f'{dir}/{mod}.jpg', bbox_inches='tight')
    plt.cla()
    plt.close(fig)

    plt.plot(list(range(len(loss_history_color))), loss_history_color, linewidth=1)
    plt.savefig(f'{dir}/_loss.jpg')
    plt.cla()
    plt.close(fig)

    plt.plot(list(range(len(loss_history_color[-100:]))), loss_history_color[-100:], linewidth=1)
    plt.savefig(f'{dir}/_loss_recent.jpg')
    plt.cla()
    plt.close(fig)



def vis_helper(step, args, eft, diffusion, vae, renderer_grid, target_cameras, target_rgb, relative_cam, loss_history, z):
    # Visualize the full renders every 100 iterations.
    '''
    target_cameras: ()
    target_rgb: (B, C, H, W) or (B, H, W, C) but we want (B, H, W, C)
    '''
    if target_rgb.shape[-1] != 3 and len(target_rgb.shape) == 4:
        target_rgb = target_rgb.permute(0,2,3,1) # (B, C, H, W)

    #! SAMPLE BATCH CAMERAS
    rand_batch = torch.randperm(len(target_cameras))
    batch_idx = rand_batch[:1]
    if args.num_input_range is not None:
        context_size = torch.randint(args.num_input_range[0], args.num_input_range[1],(1,))
    else:
        context_size = args.num_input
    render_camera, render_rgb, batch_mask, input_cameras, input_rgb, input_masks = relative_cam(target_cameras, target_rgb, context_size=context_size, query_idx=batch_idx)
    
    with torch.no_grad():

        if args.train_eft:
            eft.module.encode(input_cameras, input_rgb)
        else:
            eft.encode(input_cameras, input_rgb)

        save_visualization(
            eft,
            diffusion, 
            vae,
            args.scale_factor,
            render_camera, 
            render_rgb,
            loss_history,
            renderer_grid,
            dir=f'{args.exp_dir}/log/',
            mod=f'train_{step:06d}',
            input_rgb=input_rgb,
            input_cameras=input_cameras,
            z=z,
            args=args
        )


def normalize(x):
    return torch.clip(x*2 - 1.0, -1.0, 1.0)

def unnormalize(x):
    return torch.clip((x + 1.0) / 2.0, 0.0, 1.0)


def init_renderer(gpu, img_h, img_w, min=0.1, max=4.0, bbox=None, n_pts_per_ray=20, n_rays=750, scale_factor=None):
    '''
    Return renderers

    renderer_grid - full resolution
    renderer_feat - scaled down feature grid resolution 
    renderer_mc - random sampling
    '''
    img_h, img_w = img_h, img_w
    volume_extent_world = max
    half_pix_width = 1.0 / img_w
    half_pix_height = 1.0 / img_h

    raysampler_grid = GridRaysampler(
        min_x=1.0 - half_pix_width,
        max_x=-1.0 + half_pix_width,
        min_y=1.0 - half_pix_height,
        max_y=-1.0 + half_pix_height,
        image_height=img_h,
        image_width=img_w,
        n_pts_per_ray=n_pts_per_ray,
        min_depth=min,
        max_depth=volume_extent_world,
    )
    if scale_factor is not None:
        raysampler_features = GridRaysampler(
            min_x=1.0 - half_pix_width,
            max_x=-1.0 + half_pix_width,
            min_y=1.0 - half_pix_height,
            max_y=-1.0 + half_pix_height,
            image_height=int(img_h//scale_factor),
            image_width=int(img_w//scale_factor),
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min,
            max_depth=volume_extent_world,
        )
    if bbox is None:
        raysampler_mc = MonteCarloRaysampler(
            min_x = -1.0,
            max_x = 1.0,
            min_y = -1.0,
            max_y = 1.0,
            n_rays_per_image=n_rays,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min,
            max_depth=volume_extent_world,
        )
    elif bbox is not None:
        raysampler_mc = MonteCarloRaysampler(
            min_x = -bbox[0,1],
            max_x = -bbox[0,3],
            min_y = -bbox[0,0],
            max_y = -bbox[0,2],
            n_rays_per_image=n_rays,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min,
            max_depth=volume_extent_world,
        )

    raymarcher = LightFieldRaymarcher()

    renderer_grid = CustomImplicitRenderer(
        raysampler=raysampler_grid, raymarcher=raymarcher, reg=True
    )
    renderer_mc = CustomImplicitRenderer(
        raysampler=raysampler_mc, raymarcher=raymarcher, reg=True
    )

    renderer_grid = renderer_grid.cuda(gpu)
    renderer_mc = renderer_mc.cuda(gpu)

    if scale_factor is None:
        return renderer_grid, renderer_mc
    else:
        renderer_feat = CustomImplicitRenderer(
            raysampler=raysampler_features, raymarcher=raymarcher, reg=True
        )
        renderer_feat = renderer_feat.cuda(gpu)
        return renderer_grid, renderer_feat, renderer_mc


def load_dataset(args, image_size=256):
    '''
    Return dataset
    '''
    if args.dataset_name == 'co3d':
        train_dataset = CO3Dv2Wrapper(root=args.root, category=args.category, sample_batch_size=20, image_size=image_size)
    return train_dataset


def train(gpu, args):

    #! INIT DISTRIBUTED PROCESS
    rank = args.nr * args.gpus + gpu
    print('spawning gpu rank', rank, 'out of', args.gpus, 'using', args.backend)
    dist.init_process_group(args.backend, rank=rank, world_size=args.world_size)
    torch.cuda.set_device(gpu)
    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(args.exp_dir + '/log/', exist_ok=True)

    #! LOAD MODEL
    eft, vae, diffusion = load_model(gpu, args)

    #! OPTIMIZER
    lr = 5e-5
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.5)
    diffusion = nn.parallel.DistributedDataParallel(diffusion, device_ids=[gpu], find_unused_parameters=True)

    if args.train_eft:
        print('training eft')
        optimizer_nerf = torch.optim.Adam(eft.parameters(), lr=lr)
        scheduler_nerf = torch.optim.lr_scheduler.StepLR(optimizer_nerf, step_size=50000, gamma=0.5)
        eft = nn.parallel.DistributedDataParallel(eft, device_ids=[gpu])

    #! LOAD DATALOADER
    batch_size = 1
    print('preparing to load ds on gpu', gpu)
    train_dataset = load_dataset(args, image_size=args.image_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            pin_memory=True,
                                            sampler=train_sampler)
    print('training data len', len(train_dataset))

    #! INIT RELATIVE CAMERA LOADER
    relative_cam = RelativeCameraLoader(relative=True, center_at_origin=True)

    #! INITIALIZE TRAINING LOOP
    #! TRAIN PARAMTERS
    n_repeat = args.repeat
    max_epoch = 1000
    vis_itr=args.vis_itr
    save_itr=args.save_itr
    render_batch_size = 1

    max_step = n_repeat*max_epoch*len(train_loader)
    if gpu == 0:
        pbar = tqdm(total=max_step)
        step = 0
        pbar.n = step 
        pbar.last_print_n = step
        pbar.refresh()

    loss_history = []
    running_loss = 0
    for ep in range(max_epoch):
        for data in train_loader:

            for re in range(n_repeat):
                #! FIT CURRENT BATCH
                optimizer.zero_grad()
                if args.train_eft:
                    optimizer_nerf.zero_grad()
                torch.autograd.set_detect_anomaly(True)
                
                #! LOAD DATA
                target_cameras = PerspectiveCameras(R=data['R'][0],T=data['T'][0],focal_length=data['f'][0],principal_point=data['c'][0],image_size=data['image_size'][0]).cuda(gpu)
                target_rgb = data['images'][0].cuda(gpu)
                bbox = data['bbox'][0].cuda(gpu)
                valid_region = data['valid_region'][0].cuda(gpu)
                
                    
                #! SAMPLE BATCH CAMERAS
                rand_batch = torch.randperm(len(target_cameras))
                batch_idx = rand_batch[:render_batch_size]
                if args.num_input_range is not None:
                    context_size = torch.randint(args.num_input_range[0], args.num_input_range[1],(1,))
                else:
                    context_size = args.num_input
                batch_cameras, batch_rgb, batch_mask, input_cameras, input_rgb, input_masks, context_idx = relative_cam(target_cameras, target_rgb, context_size=context_size, query_idx=batch_idx, return_context=True)
                input_bbox = bbox[context_idx]
                batch_valid_region = valid_region[batch_idx]

                #! INITIALIZE RENDERERS
                if args.dataset_name == 'co3d':
                    cam_dist_mean = torch.mean(torch.linalg.norm(target_cameras.get_camera_center(), axis=1))
                    min_depth = cam_dist_mean - 5.0
                    volume_extent_world = cam_dist_mean + 5.0
                    if bbox is None:
                        renderer_grid, renderer_feat, renderer_mc = init_renderer(gpu, train_dataset.img_h, train_dataset.img_w, min=min_depth, max=volume_extent_world, scale_factor=args.scale_factor)
                    elif bbox is not None:
                        renderer_grid, renderer_feat, renderer_mc = init_renderer(gpu, train_dataset.img_h, train_dataset.img_w, min=min_depth, max=volume_extent_world, bbox=bbox[batch_idx], scale_factor=args.scale_factor)

                #! EFT
                if args.train_eft:
                    epipolar_features, sampled_rays, reg_term = renderer_feat(
                        cameras=batch_cameras, 
                        volumetric_function=eft.module.batched_forward,
                        n_batches=1,
                        input_cameras=input_cameras,
                        input_rgb=input_rgb
                    )
                else:
                    with torch.no_grad():
                        epipolar_features, sampled_rays, reg_term = renderer_feat(
                            cameras=batch_cameras, 
                            volumetric_function=eft.batched_forward,
                            n_batches=1,
                            input_cameras=input_cameras,
                            input_rgb=input_rgb
                        )
                    
                rendered_images, epipolar_latents = (
                    epipolar_features.split([3, 256], dim=-1)
                )

                #! RESHAPE
                diffusion_batch_size = args.diffusion_batch_size
                if batch_rgb.shape[1] != 3:
                    batch_rgb = rearrange(batch_rgb, 'b h w c -> b c h w')
                if args.dataset_name == 'srn':
                    batch_rgb = F.interpolate(batch_rgb, scale_factor=2.0)
                with torch.no_grad():
                    images_z = vae.encode(normalize(batch_rgb)).mode() * args.z_scale_factor

                
                epipolar_latents = rearrange(epipolar_latents, 'b h w f -> b f h w')
                
                diffusion_input = images_z.expand(diffusion_batch_size, -1, -1, -1)
                diffusion_cond_image = epipolar_latents.expand(diffusion_batch_size, -1, -1, -1)

                #! MASK LOSS OUTSIDE OF VALID REGION
                batch_mask = F.interpolate(batch_valid_region, scale_factor=0.125, mode='bilinear')
                batch_mask = batch_mask.expand(diffusion_batch_size, images_z.shape[1], -1, -1)
                batch_mask[batch_mask > 0.6] = 1.0
                batch_mask[batch_mask <= 0.6] = 0.0

                color_loss = 0
                if args.train_eft:
                    #! COMPUTE COLOR ERROR
                    colors_at_rays = sample_images_at_mc_locs(
                        batch_rgb, 
                        sampled_rays.xys
                    )

                    color_loss = huber(
                        rendered_images, 
                        colors_at_rays,
                    )

                    color_loss = color_loss * rearrange(batch_mask[:1, :1, ...], 'b c h w -> b h w c')
                    color_loss = color_loss.abs().mean()

                #! DIFFUSION ERROR
                d_loss = diffusion(diffusion_input, cond_images=diffusion_cond_image, loss_mask=batch_mask)

                #! LOSS TERM
                loss = d_loss + color_loss


                #! BACKWARD
                #! CATCH SOME ERRORS
                try:
                    loss.backward()
                    skip_save = False
                except RuntimeError:
                    print('RUNTIME ERROR')
                    print(data['idx'])
                    return
                
                #! OPTIM STEP
                optimizer.step()
                scheduler.step()
                if args.train_eft:
                    optimizer_nerf.step()
                    scheduler_nerf.step()

                if gpu == 0:
                    step += 1
                    pbar.update(1)
                    pbar.set_description(f'epoch={ep}/{max_epoch} lr={float(scheduler.get_last_lr()[0]):1.2e} loss={loss.item():02f}')
                    running_loss += loss.item()

                    #! UPDATE LR AND LOG LOSS
                    if step % 50 == 0:
                        loss_history.append(float(running_loss)/100)
                        running_loss = 0

                    #! VISUALIZE
                    if step % vis_itr == 0:
                        print('visualizing', args.exp_name, args.category)
                        vis_helper(step, args, eft, diffusion, vae, renderer_feat, target_cameras, target_rgb, relative_cam, loss_history, z=images_z)

                    #! SAVE MODEL
                    if step % save_itr == 0 and step!=0 and not skip_save:
                        save_model(args, step, f'latest', diffusion, optimizer, scheduler, eft)
                        if step % 50000 == 0:
                            save_model(args, step, f'{step:06d}', diffusion, optimizer, scheduler, eft)
                            # save_model(args, step, f'prev', diffusion, optimizer, scheduler, eft)
                        print('saving model at step', step)


def save_model(args, step, mod, model, optimizer, scheduler, eft=None):
    '''
    Saves Models
    '''
    torch.save({
        'step': step,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        }, f'{args.exp_dir}/ckpt_{mod}.pt')
    if args.train_eft:
        torch.save({
        'step': step,
        'model_state_dict': eft.module.state_dict(),
        }, f'{args.exp_dir}/ckpt_{mod}_eft.pt')


def load_model(gpu, args):
    '''
    Loads Models
    '''
    #! LOAD EFT (eft)
    eft = EpipolarFeatureTransformer(use_r=args.use_r, encoder=args.encoder, return_features=True, remove_unused_layers=False).cuda(gpu)

    if args.ckpt_path is not None:
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        eft.load_state_dict(checkpoint['model_state_dict'])
    print('LOADING 1/3 loaded eft checkpoint from', args.ckpt_path)

    #! LOAD VAE
    vae = load_vae(args.ckpt_path_vae).cuda(gpu)
    print('LOADING 2/3 loaded sd vae from', args.ckpt_path_vae)

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
    total_params = sum(p.numel() for p in unet1.parameters())
    print(f"{unet1.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    
    #! LOAD DIFFUSION
    print('using diffusion objective', args.objective)
    diffusion = DDPM(
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

    if args.ckpt_path_diffusion is not None:
        checkpoint = torch.load(args.ckpt_path_diffusion, map_location='cpu')
        diffusion.load_state_dict(checkpoint['model_state_dict'])
        print('LOADING 3/3 loaded diffusion from', args.ckpt_path_diffusion)
    else:
        print('LOADING 3/3 loaded diffusion from', 'scratch')

    return eft, vae, diffusion


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
    parser.add_argument('-c', '--category', type=str, metavar='s', required=True,
                        help='category')
    parser.add_argument('-r', '--root', type=str, default='/grogu/datasets/co3d/', metavar='s',
                        help='location of test features')
    parser.add_argument('-d', '--dataset', type=str, default='co3d', metavar='s',
                        help='dataset name')
    parser.add_argument('-b', '--backend', type=str, default='nccl', metavar='s',
                        help='nccl')
    parser.add_argument('-a', '--vae', type=str, default='checkpoints/sd/sd-v1-3-vae.ckpt', metavar='S',
                        help='vae ckpt')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'1234{args.port}'
    print('using port', f'1234{args.port}')

    torch.manual_seed(1)

    if args.dataset == 'co3d':

        #! ####################################################################
        #! TODO:  
        #! ####################################################################

        #! EXPERIMENT GROUP FOLDER
        args.exp_group = 'sparsefusion_exp'

        #! EXPERIMENT NAME
        args.exp_name = 'demo_train'

        #! EXPERIMENT MODIFIDER 
        args.mod = '_joint'
    
        #! EXPERIMENT DIRECTORY
        args.exp_dir = f'output/{args.exp_group}/{args.exp_name}/{args.category}{args.mod}'

        #! STABLE DIFFUSION VAE PATH
        args.ckpt_path_vae = args.vae

        #! NUMBER OF INPUTS VIEWS
        args.num_input_range = (2,6) # [low, high) 

        #! ####################################################################
        #! ABOVE: TODO 
        #! ####################################################################

        #@ AUTOMATIC RESUME
        #! WARNING: CURRENTLY DOES NOT RESUME OPTIMIZER
        if os.path.exists(f'{args.exp_dir}/ckpt_latest_eft.pt'):
            print('***automatically resuming***')
            args.ckpt_path = f'{args.exp_dir}/ckpt_latest_eft.pt'
            args.ckpt_path_diffusion = f'{args.exp_dir}/ckpt_latest.pt'
        else:
            print('***training from scratch***')
            args.ckpt_path = None # or default pretrained
            args.ckpt_path_diffusion = None # or default pretrained

        #@ SET PARAMETERS
        args.timesteps = 500                # default diffusion steps (500)
        args.objective = 'noise'            # default objective (noise)
        args.scale_factor = 8               # default VAE scale factor
        args.image_size = 256               # default image resolution
        args.dataset_name = 'co3d'          # dataset name
        args.diffusion_batch_size = 12      # extend batch size (12)
        args.repeat = 1                     # repeat a scene (1)
        args.vis_itr = 100                  # default visualization freq (100)
        args.save_itr = 1000                # save iteration
        args.train_eft = True               # jointly train EFT (eft)
        if args.train_eft:
            print('joint eft training')
        

    # Modification below not recommended 
    args.use_r = True
    args.encoder = 'resnet18'
    args.num_input = 4
    args.z_scale_factor = 0.18215
    
    mp.spawn(train, nprocs=args.gpus, args=(args,))

if __name__ == '__main__':
    os.environ["TORCH_DISTRIBUTED_DEBUG"]='INFO'
    main()