'''
Initialize Renderers
'''

import torch
from pytorch3d.renderer import (
    PerspectiveCameras,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    GridRaysampler,
)
from utils.eft_renderer import CustomImplicitRenderer
from utils.eft_raymarcher import LightFieldRaymarcher


def init_ray_sampler(gpu, img_h, img_w, min=0.1, max=4.0, bbox=None, n_pts_per_ray=128, n_rays=750, scale_factor=None):
    '''
    Construct ray samplers for torch-ngp

    Args:
        gpu (int): gpu id
        img_h (int): image height
        img_w (int): image width
        min (int): min depth for point along ray
        max (int): max depth for point along ray
        bbox (List): bounding box for monte carlo sampler
        n_pts_per_ray (int): number of points along a ray
        n_rays (int): number of rays for monte carlo sampler
        scale_factor (int): return a grid sampler at a scale factor
    
    Returns:
        sampler_grid (sampler): a grid sampler at full resolution
        sampler_mc (sampler): a monte carlo sampler
        sampler_feat (sampler): a grid sampler at scale factor resolution
            if scale factor is provided
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
            n_pts_per_ray=20,
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

    if scale_factor is not None:
        return raysampler_grid, raysampler_mc, raysampler_features
    else:
        return raysampler_grid, raysampler_mc


def init_light_field_renderer(gpu, img_h, img_w, min=0.1, max=4.0, bbox=None, n_pts_per_ray=128, n_rays=750, scale_factor=None):
    '''
    Construct implicit renderers for EFT

    Args:
        gpu (int): gpu id
        img_h (int): image height
        img_w (int): image width
        min (int): min depth for point along ray
        max (int): max depth for point along ray
        bbox (List): bounding box for monte carlo sampler
        n_pts_per_ray (int): number of points along a ray
        n_rays (int): number of rays for monte carlo sampler
        scale_factor (int): return a grid sampler at a scale factor
    
    Returns:
        renderer_grid (renderer): a grid renderer at full resolution
        renderer_mc (renderer): a monte carlo renderer
        renderer_feat (renderer): a grid renderer at scale factor resolution
            if scale factor is provided
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
            n_pts_per_ray=20,
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
        return renderer_grid, renderer_mc, renderer_feat