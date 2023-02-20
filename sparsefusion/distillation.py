'''
Diffusion distillation loop
'''
import numpy as np
import imageio
import time
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from pytorch3d.renderer import PerspectiveCameras
from einops import rearrange, reduce, repeat

from external.nerf.network_grid import NeRFNetwork
from external.plms import PLMSSampler
from external.external_utils import PerceptualLoss
from utils.camera_utils import RelativeCameraLoader, get_interpolated_path
from utils.common_utils import get_lpips_fn, get_metrics, split_list, normalize, unnormalize, huber
from utils.co3d_dataloader import CO3Dv2Wrapper
from utils.co3d_dataloader import CO3D_ALL_CATEGORIES, CO3D_ALL_TEN
from utils.render_utils import init_ray_sampler, init_light_field_renderer

def distillation_loop(
        gpu,
        args,
        opt,
        model_tuple,
        save_dir,
        seq_name,
        scene_cameras,
        scene_rgb,
        scene_mask,
        scene_valid_region,
        input_idx,
        use_diffusion=True,
        max_itr=3000,
        loss_fn_vgg=None,
    ):
    '''
    Loop for diffusion distillation
    Saves optimized torch-ngp

    Args:
        gpu (int): gpu id
        args (Namespace): SparseFusion options
        opt (Namesapce): torch-ngp options
        model_tuple (EFT, VAE, VLDM): a tuple of three models
        save_dir (str): save directory
        seq_name (str): save sequence name
        scene_cameras (PyTorch3D Camera): cameras
        scene_rgb (Tensor): gt rgb
        scene_mask (Tensor): foreground mask
        scene_valid_region (Tensor): valid image region 

    '''

    os.makedirs(f'{args.exp_dir}/render_imgs/{seq_name}/', exist_ok=True)
    os.makedirs(f'{args.exp_dir}/render_gifs/', exist_ok=True)
    eft, vae, vldm = model_tuple

    #@ GET RELATIVE CAMERA LOADER
    relative_cam = RelativeCameraLoader(relative=True, center_at_origin=True)
    relative_cam_no_origin = RelativeCameraLoader(relative=True, center_at_origin=False)

    #@ GET RELATIVE CAMERAS     
    scene_cameras_rel = relative_cam.get_relative_camera(scene_cameras, query_idx=[0], center_at_origin=True)
    scene_cameras_vox = relative_cam_no_origin.get_relative_camera(scene_cameras, query_idx=[0], center_at_origin=False)

    #@ GET ADDITIONAL CAMERAS
    scene_cameras_aug = get_interpolated_path(scene_cameras, n=50, method='circle', theta_offset_max=0.17)
    scene_cameras_aug = relative_cam.concat_cameras([scene_cameras, scene_cameras_aug])
    scene_cameras_aug_rel = relative_cam.get_relative_camera(scene_cameras_aug, query_idx=[0], center_at_origin=True)
    scene_cameras_aug_vox = relative_cam_no_origin.get_relative_camera(scene_cameras_aug, query_idx=[0], center_at_origin=False)
    blank_rgb = torch.zeros_like(scene_rgb[:1])
    blank_rgb = blank_rgb.repeat(len(scene_cameras_aug), 1, 1, 1)
    scene_rgb_aug = torch.cat((scene_rgb, blank_rgb))

    #@ ADJUST RENDERERS
    cam_dist_mean = torch.mean(torch.linalg.norm(scene_cameras.get_camera_center(), axis=1))
    min_depth = cam_dist_mean - 5.0
    volume_extent_world = cam_dist_mean + 5.0
    sampler_grid, _, sampler_feat = init_ray_sampler(gpu, 256, 256, min=min_depth, max=volume_extent_world, scale_factor=2)
    _, _, renderer_feat = init_light_field_renderer(gpu, 256, 256, min=min_depth, max=volume_extent_world, scale_factor=8.0)
    
    #! ###############################
    #! ####### PREPROCESSING #########
    #! ###############################
    #@ CACHE VIEW CONDITIONED FEATURES
    if use_diffusion:
        eft_feature_cache = {}
        timer = time.time()
        for ci in tqdm(range(len(scene_cameras_aug_rel))):

            #@ GET EFT REL CAMERAS
            gpnr_render_camera, render_rgb, batch_mask, input_cameras, input_rgb, input_masks = relative_cam(scene_cameras_aug_rel, scene_rgb_aug, query_idx=[ci], context_idx=input_idx)
            eft.encode(input_cameras, input_rgb)

            #@ GET EFT FEATURES ANDS IMAGE
            with torch.no_grad():
                epipolar_features , _, _ = renderer_feat(
                    cameras=gpnr_render_camera, 
                    volumetric_function=eft.batched_forward,
                    n_batches=16,
                    input_cameras=input_cameras,
                    input_rgb=input_rgb
                )
                lr_render_, epipolar_latents = (
                    epipolar_features.split([3, 256], dim=-1)
                )

            query_camera = relative_cam.get_camera_slice(scene_cameras_aug_rel, [ci])
            query_camera_vox = relative_cam.get_camera_slice(scene_cameras_aug_vox, [ci])
            epipolar_latents = rearrange(epipolar_latents, 'b h w f -> b f h w')

            lr_image = rearrange(lr_render_, 'b h w f -> b f h w')
            lr_image = F.interpolate(lr_image, scale_factor=8.0, mode='bilinear')
            
            #@ OPTIONAL DIFFUSION IMAGE
            diffusion_image = None

            eft_feature_cache[ci] = {'camera':query_camera, 'camera_vox':query_camera_vox, 'features':epipolar_latents, 
                                        'diffusion_image':diffusion_image, 'eft_image':lr_image}
            
        print(f'cached {len(eft_feature_cache)} features in {(time.time() - timer):02f} seconds')
        
        if len(eft_feature_cache) >= len(scene_cameras_rel):
            n_per_row = 8
            vis_rows = []
            for i in range(0, len(scene_cameras_rel), n_per_row):
                temp_row = []
                for j in range(n_per_row):
                    img = eft_feature_cache[i + j]['eft_image']
                    vis_img = rearrange(img, 'b c h w -> b h w c')[0].detach().cpu().numpy()
                    temp_row.append(vis_img)
                temp_row = np.hstack((temp_row))
                vis_rows.append(temp_row)
            vis_grid = np.vstack(vis_rows)
            imageio.imwrite(f'{args.exp_dir}/log/{seq_name}_eft_grid.jpg', (vis_grid*255).astype(np.uint8))

    #! ###############################
    #! ###### SET UP TORCH NGP #######
    #! ###############################

    #@ REGULARIZATION HYPERPARAMETERS
    lambda_color = 1.0 #1.0
    lambda_sil = 1.0
    lambda_opacity = 1e-3
    lambda_entropy = 0.0
    lambda_percep = 0.0

    render_batch_size = 1
    max_itr = max_itr 
    start_fusion_step = 1000
    start_percep_step = 1000

    #@ DEFINE HELPER MODULES
    plms_sampler = PLMSSampler(vldm, 50)
    perceptual_loss = PerceptualLoss('vgg', device=f'cuda:{gpu}')

    #@ INITIALIZE TORCH NGP
    ngp_network = NeRFNetwork(opt).cuda(gpu).train()
    optimizer = torch.optim.Adam(ngp_network.get_params(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.2)
    running_loss = 0.0
    loss_list, fusion_loss_list, opacity_loss_list, entropy_loss_list = [], [], [], []

    #! ###############################
    #! ########## MAIN LOOP ##########
    #! ###############################

    for itr in tqdm(range(max_itr)):

        if itr == start_percep_step:
            print('turning on percep loss')
            lambda_percep = 0.1

        ngp_network.train()
        if opt.cuda_ray and itr % 16 == 0:
            ngp_network.update_extra_state()
        
        #@ SAMPLE BATCH
        rand_batch = torch.randperm(len(input_idx))
        batch_idx = input_idx[rand_batch[:render_batch_size]]
        assert batch_idx in input_idx
        if render_batch_size == 1:
            batch_idx = [batch_idx]
        batch_cameras = relative_cam.get_camera_slice(scene_cameras_vox, batch_idx)
        batch_rgb = scene_rgb[batch_idx]
        batch_valid_region = scene_valid_region[batch_idx]
        batch_valid_region = torch.ones_like(batch_valid_region)
        if scene_mask is not None:
            batch_mask = scene_mask[batch_idx]

        #@ FETCH RAYS
        ray_bundle = sampler_feat(batch_cameras)
        H, W = ray_bundle.origins.shape[1], ray_bundle.origins.shape[2]
        rays_o = rearrange(ray_bundle.origins, 'b h w c -> b (h w) c') # [B, N, 3]
        rays_d = rearrange(ray_bundle.directions, 'b h w c -> b (h w) c') # [B, N, 3]
        B, N = rays_o.shape[:2]                    

        #@ RENDER
        bg_color = 0
        outputs = ngp_network.render(rays_o, rays_d, staged=False, perturb=True, bg_color=bg_color, ambient_ratio=1.0, shading='albedo', force_all_rays=True, **vars(opt))
        rendered_images = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]
        rendered_silhouettes = outputs['weights_sum'].reshape(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
        #' rendered_images (B, 3, H, W)


        #@ COMPUTE COLOR LOSS
        batch_rgb = F.interpolate(batch_rgb, scale_factor=1.0/opt.hw_scale)
        batch_mask = F.interpolate(batch_mask, scale_factor=1.0/opt.hw_scale)
        color_err = (huber(rendered_images, batch_rgb)).abs().mean()

        #@ COMPUTE MASK LOSS
        if scene_mask is not None:
            sil_err = (huber(rendered_silhouettes, batch_mask)).abs().mean()
        else:
            sil_err = 0
        
        #@ VOLUMETRIC LOSS
        loss = lambda_color * color_err + lambda_sil * sil_err

        #@ REGULARIZATIONS
        #@ OPACITY
        opacity_term = torch.zeros_like(color_err)
        if lambda_opacity > 0:
            opacity_term = torch.sqrt((rendered_silhouettes ** 2) + .01).mean()
            loss += lambda_opacity * opacity_term

        #@ ENTROPY
        entropy_term = torch.zeros_like(color_err)
        if lambda_entropy > 0:
            alphas = (rendered_silhouettes).clamp(1e-5, 1 - 1e-5)
            entropy_term = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
            loss += lambda_entropy * entropy_term

        #@ BACKPROP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        if itr % 10 == 0 and itr != 0:
            running_loss = 0

        #@ LOG LOSS
        loss_list.append(loss.item())
        opacity_loss_list.append((lambda_opacity * opacity_term).item())
        entropy_loss_list.append((lambda_entropy * entropy_term).item())

        #@ FUSION
        if use_diffusion:
            
            optimizer.zero_grad()
            #@ SAMPLE CAMERAS
            rand_batch = torch.randperm(len(eft_feature_cache))
            batch_idx = rand_batch[1]
            batch_cached = eft_feature_cache[int(batch_idx)]
            batch_cameras = batch_cached['camera_vox']
            batch_features = batch_cached['features'].detach()
            batch_noisy_rgb = batch_cached['eft_image'].detach()
            batch_noisy_mask = batch_noisy_rgb.mean(dim=1, keepdim=True)
            batch_noisy_mask[batch_noisy_mask > .1] = 1
            batch_noisy_mask[batch_noisy_mask <= .1] = 0

            #@ FETCH RAYS
            ray_bundle = sampler_feat(batch_cameras)
            H, W = ray_bundle.origins.shape[1], ray_bundle.origins.shape[2]
            rays_o = rearrange(ray_bundle.origins, 'b h w c -> b (h w) c') # [B, N, 3]
            rays_d = rearrange(ray_bundle.directions, 'b h w c -> b (h w) c') # [B, N, 3]
            B, N = rays_o.shape[:2]                    

            #@ RENDER
            bg_color = 0
            outputs = ngp_network.render(rays_o, rays_d, staged=False, perturb=True, bg_color=bg_color, ambient_ratio=1.0, shading='albedo', force_all_rays=True, **vars(opt))
            rendered_images = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]
            rendered_silhouettes = outputs['weights_sum'].reshape(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
            #' rendered_images (B, 3, H, W)

            rendered_images = F.interpolate(rendered_images, scale_factor=opt.hw_scale, mode='bilinear')
            rendered_silhouettes = F.interpolate(rendered_silhouettes, scale_factor=opt.hw_scale, mode='bilinear')
            

            #! ###############################
            #! ###### DISTILLATION STEP ######
            #! ###############################
            if itr > start_fusion_step:
                #@ DIFFUSION DISTILLATION

                #@ ENCODE RENDERING
                with torch.no_grad():
                    latents = vae.encode(normalize(rendered_images)).mode() * args.z_scale_factor

                #@ FAST DIFFUSION SAMPLING
                with torch.no_grad():
                    max_thres = torch.rand((1)).clamp(min=0.0, max=0.99).item()
                    pred_x0, x_noisy, noise, alpha_cumprod = plms_sampler.sample(latents, cond_images=batch_features.expand(-1,-1,-1,-1), use_tqdm=False, return_noise=True, max_thres=max_thres)
            
                #@ PIXEL SPACE LOSS
                fusion_weight = (1 - alpha_cumprod).to(pred_x0.device)
                with torch.no_grad():
                    pred_img = unnormalize(vae.decode(1.0 / args.z_scale_factor * pred_x0)).clip(0.0, 1.0)
                fusion_loss = fusion_weight * (rendered_images - pred_img).abs().mean()

                if lambda_percep > 0.0:
                    percep_term = perceptual_loss(rendered_images, pred_img, normalize=True)
                    fusion_loss += percep_term.mean() * lambda_percep

            else:
                #@ EFT BOOTSTRAP

                #@ COMPUTE COLOR LOSS
                color_err = (huber(rendered_images, batch_noisy_rgb)).abs().mean()

                #@ COMPUTE MASK LOSS
                if scene_mask is not None:
                    sil_err = (huber(rendered_silhouettes, batch_noisy_mask)).abs().mean()
                else:
                    sil_err = 0
                
                #@ RENDERING LOSS
                fusion_loss = lambda_color * color_err + lambda_sil * sil_err

            #@ REGULARIZATIONS

            #@ OPACITY
            opacity_term = torch.zeros((1)).cuda(gpu).requires_grad_()
            if lambda_opacity > 0:
                opacity_term = torch.sqrt((rendered_silhouettes ** 2) + .01).mean()

            #@ ENTROPY
            entropy_term = torch.zeros((1)).cuda(gpu).requires_grad_()
            if lambda_entropy > 0:
                alphas = (rendered_silhouettes).clamp(1e-5, 1 - 1e-5)
                entropy_term = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
            
            loss = fusion_loss + lambda_opacity * opacity_term + lambda_entropy * entropy_term
            loss.backward()


            #@ LOG
            fusion_loss_list.append(fusion_loss.item())

            #@ STEP W
            optimizer.step()

        #@ SAVE INTERMEDIATE IMAGE
        if itr % 20 == 0:
            plt.plot(list(range(len(loss_list))), loss_list, linewidth=1, label="volumetric")
            plt.plot(list(range(len(opacity_loss_list))), opacity_loss_list, linewidth=1, label="opacity")
            plt.plot(list(range(len(entropy_loss_list))), entropy_loss_list, linewidth=1, label="entropy")
            plt.legend(loc="upper right")
            plt.savefig(f'{args.exp_dir}/log/{seq_name}_loss.jpg')
            plt.cla()
            plt.close()

            plt.plot(list(range(len(fusion_loss_list))), fusion_loss_list, linewidth=1)
            plt.savefig(f'{args.exp_dir}/log/{seq_name}_fusionloss.jpg')
            plt.cla()
            plt.close()

            with torch.no_grad():
                #@ FETCH RAYS
                ngp_network.eval()
                ray_bundle = sampler_grid(batch_cameras)
                H, W = ray_bundle.origins.shape[1], ray_bundle.origins.shape[2]
                rays_o = rearrange(ray_bundle.origins, 'b h w c -> b (h w) c') # [B, N, 3]
                rays_d = rearrange(ray_bundle.directions, 'b h w c -> b (h w) c') # [B, N, 3]
                B, N = rays_o.shape[:2]                    

                #@ RENDER
                bg_color = 0
                outputs = ngp_network.render_batched(rays_o, rays_d, batched=True, perturb=True, bg_color=bg_color, ambient_ratio=1.0, shading='albedo', force_all_rays=True, **vars(opt))
                rendered_images = outputs['image'].reshape(B, H, W, 3).contiguous()[0] # [1, 3, H, W]
                rendered_silhouettes = outputs['weights_sum'].reshape(B, H, W, 1).contiguous()[0]

            #' rendered_image (256, 256, 3)
            rendered_image_vis = rendered_images.detach().cpu().numpy()
            rendered_sil_vis = rendered_silhouettes.expand(-1,-1,3).detach().cpu().numpy()
            render_vis = np.hstack((rendered_image_vis, rendered_sil_vis))
            imageio.imwrite(f'{args.exp_dir}/log/{seq_name}_vis.jpg', (render_vis*255).astype(np.uint8))


    #@ SEQUENCE GIF AND METRIC
    print('rendering sequence', end='')
    seq_rgb_list, seq_sil_list, gt_rgb_list, gt_sil_list, ldm_rgb_list = [], [], [], [], []
    scene_psnr, scene_ssim, scene_lp = [], [], []
    for ci in range(len(scene_cameras_vox)):
        render_camera = relative_cam.get_camera_slice(scene_cameras_vox, [ci])
        target_rgb = scene_rgb[[ci]]
        target_sil = scene_mask[[ci]]

        with torch.no_grad():
            #@ FETCH RAYS
            ngp_network.eval()
            ray_bundle = sampler_grid(render_camera)
            H, W = ray_bundle.origins.shape[1], ray_bundle.origins.shape[2]
            rays_o = rearrange(ray_bundle.origins, 'b h w c -> b (h w) c') # [B, N, 3]
            rays_d = rearrange(ray_bundle.directions, 'b h w c -> b (h w) c') # [B, N, 3]
            B, N = rays_o.shape[:2]                    

            #@ RENDER
            bg_color = 0
            outputs = ngp_network.render_batched(rays_o, rays_d, batched=True, perturb=True, bg_color=bg_color, ambient_ratio=1.0, shading='albedo', force_all_rays=True, **vars(opt))
            rendered_images = outputs['image'].reshape(B, H, W, 3) # [1, H, W, 3]
            rendered_silhouettes = outputs['weights_sum'].reshape(B, H, W, 1) # [1, H, W, 3]
            rendered_images = rendered_images[0].detach().cpu().numpy()
            rendered_silhouettes = rendered_silhouettes.expand(-1,-1,-1,3)[0].detach().cpu().numpy()

        gt = rearrange(target_rgb, 'b c h w -> b h w c')[0].detach().cpu().numpy()
        gt_sil = rearrange(target_sil.expand(-1,3,-1,-1), 'b c h w -> b h w c')[0].detach().cpu().numpy()
        seq_rgb_list.append(rendered_images)
        seq_sil_list.append(rendered_silhouettes)
        gt_rgb_list.append(gt)
        gt_sil_list.append(gt_sil)

        if use_diffusion:
            ldm_images = rearrange(eft_feature_cache[ci]['eft_image'], 'b c h w -> b h w c')[0].detach().cpu().numpy()
            ldm_rgb_list.append(ldm_images)

        s, p, l = get_metrics(rendered_images, gt, use_lpips=True, loss_fn_vgg=loss_fn_vgg)
        scene_psnr.append(p)
        scene_ssim.append(s)
        scene_lp.append(l)
        print('.', end='', flush=True)
    print('')
    print('warning: this metric is used for debugging only and not the final metric')
    print(f'{args.category} scene {seq_name}')
    print('psnr:', np.array(scene_psnr).mean())
    print('lpips:', np.array(scene_lp).mean())

    with open(f'{save_dir}/metrics/{seq_name}.txt', 'w') as fp:
        fp.write(f'warning: this metric is used for debugging only and not the final metric')
        fp.write(f'psnr:\n' + str(np.array(scene_psnr).mean()) + '\n')
        fp.write(f'ssim:\n' + str(np.array(scene_ssim).mean()) + '\n')
        fp.write(f'pip:\n' + str(np.array(scene_lp).mean()) + '\n')

    gif_path = f'{save_dir}/render_gifs/{seq_name}.gif'
    seq_dir = f'{save_dir}/render_imgs/{seq_name}'
    os.makedirs(seq_dir, exist_ok=True)
    with imageio.get_writer(gif_path, mode='I', duration=0.2) as writer:
        for si in range(len(seq_rgb_list)):
            if use_diffusion:
                writer.append_data((np.hstack((gt_rgb_list[si], ldm_rgb_list[si], seq_rgb_list[si], gt_sil_list[si], seq_sil_list[si]))*255).astype(np.uint8))
            else:
                writer.append_data((np.hstack((gt_rgb_list[si], seq_rgb_list[si], gt_sil_list[si], seq_sil_list[si]))*255).astype(np.uint8))

            frame_vis = np.hstack((gt_rgb_list[si], seq_rgb_list[si]))
            imageio.imwrite(f'{seq_dir}/{si:03d}.jpg', (frame_vis*255).astype(np.uint8))
    print('saved video', gif_path)
        

    #@ VISUALIZE WITH CIRCLE
    print('rendering circle', end='')
    circle_rgb_list, circle_sil_list = [], []
    circle_cameras = get_interpolated_path(scene_cameras_vox, n=50, method='circle')
    for ci in range(len(circle_cameras)):
        render_camera = relative_cam.get_camera_slice(circle_cameras, [ci])
        with torch.no_grad():
            #@ FETCH RAYS
            ngp_network.eval()
            ray_bundle = sampler_grid(render_camera)
            H, W = ray_bundle.origins.shape[1], ray_bundle.origins.shape[2]
            rays_o = rearrange(ray_bundle.origins, 'b h w c -> b (h w) c') # [B, N, 3]
            rays_d = rearrange(ray_bundle.directions, 'b h w c -> b (h w) c') # [B, N, 3]
            B, N = rays_o.shape[:2]                    

            #@ RENDER
            bg_color = 0
            outputs = ngp_network.render_batched(rays_o, rays_d, batched=True, perturb=True, bg_color=bg_color, ambient_ratio=1.0, shading='albedo', force_all_rays=True, **vars(opt))
            rendered_images = outputs['image'].reshape(B, H, W, 3) # [1, H, W, 3]
            rendered_silhouettes = outputs['weights_sum'].reshape(B, H, W, 1) # [1, H, W, 3]
            rendered_images = rendered_images[0].detach().cpu().numpy()
            rendered_silhouettes = rendered_silhouettes.expand(-1,-1,-1,3)[0].detach().cpu().numpy()

        circle_rgb_list.append(rendered_images)
        circle_sil_list.append(rendered_silhouettes)
        print('.', end='', flush=True)
    print('')
    gif_path = f'{save_dir}/render_gifs/{seq_name}_circle.gif'
    with imageio.get_writer(gif_path, mode='I', duration=0.2) as writer:
        for si in range(len(circle_rgb_list)):
            writer.append_data((np.hstack((circle_rgb_list[si], circle_sil_list[si]))*255).astype(np.uint8))
    print('saved video', gif_path)
    

    #@ SAVE NGP MODEL
    w_addr = f'{save_dir}/{seq_name}.pt'
    torch.save({'model_state_dict': ngp_network.state_dict()}, w_addr)
    print('input idx', input_idx)


def get_default_torch_ngp_opt():
    '''
    Return default options for torch-ngp
    '''
    opt = argparse.Namespace()
    opt.cuda_ray = False
    opt.max_steps = 256
    opt.num_steps = 64
    opt.upsample_steps = 64
    opt.update_extra_interval = 16
    opt.max_ray_batch = 4096
    opt.albedo_iters = 1000
    opt.bg_radius = 0
    opt.density_thresh = 10
    opt.fp16 = True
    opt.backbone = 'grid'
    opt.w = 128
    opt.h = 128
    opt.hw_scale = 2
    opt.bound = 4
    opt.min_near = 0.1
    opt.dt_gamma = 0
    opt.lambda_entropy = 1e-4
    opt.lambda_opacity = 0
    opt.lambda_orient = 1e-2
    opt.lambda_smooth = 0
    return opt