'''
Class for Epipolar Feature Extractor EFT
#@ Based upon https://github.com/google-research/google-research/blob/master/gen_patch_neural_rendering/
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.implicit.utils import (
    _validate_ray_bundle_variables, ray_bundle_variables_to_ray_points)

from utils.common_utils import HarmonicEmbedding


class TransformerEncoder(nn.Module):
    def __init__(self, d_in, d_out, n_hidden=256, n_layer=4, post_linear=False):
        super().__init__()

        self.post_linear = post_linear

        encoder_layer = nn.TransformerEncoderLayer(n_hidden, 1, n_hidden, 0.1)
        self.pre = nn.Sequential(
            nn.Linear(d_in, n_hidden),
            nn.GELU(),
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layer)
        if self.post_linear:
            self.post = nn.Sequential(nn.Linear(n_hidden, d_out))
            self.post[0].bias.data.zero_()
            self.post[0].weight.data.uniform_(-.1,.1)

    def forward(self, w, pos=None, mask=None):
        """
        Optimize given volume of latent code z
        Args:
            z: volume of latent codes (B, N_CUBES, N_HIDDEN)
            cube_xyz: xyz embedding of cube centers (B, N_CUBES, N_HARMONIC_XYZ)
        Returns:
            z: volume of latent codes (B, N_CUBES, N_HIDDEN)
        """
        if pos is not None:
            w = torch.cat((w, pos), dim=-1)
        out = self.pre(w)
        out = self.encoder(out, mask=mask)
        if self.post_linear:
            out = self.post(out)
        return out


class EpipolarFeatureTransformer(torch.nn.Module):
    def __init__(self,
                use_r=True,
                 n_harmonic_functions=6,
                 conv_dims=[32,],
                 return_features=False,
                 encoder='lite',
                 remove_unused_layers=True,
                 in_dim=3,
                 out_dim=3,
                 out_sigmoid=True,
                 omega0 = 1.0,
                 verbose=False,
                ):
        super().__init__()
        """
        Args:
            
        """
        self.use_r = use_r
        self.return_features = return_features
        self.encoder = encoder
        self.in_dim = in_dim

        if verbose:
            print('use reference r:', self.use_r)

        #! DEFINE HARMONIC EMBEDDING
        if verbose:
            print('using omega0', omega0)
        self.omega0 = omega0
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions, self.omega0)
        
        #! CNN ENCODER
        if self.encoder == 'lite':
            self.conv_dims = conv_dims
            conv_list = [nn.Conv2d(3, conv_dims[0], kernel_size=11, stride=1, padding=5), nn.ELU()]
            for conv_i, conv_dim in enumerate(conv_dims[1:]):
                conv_list.append(nn.Conv2d(conv_dims[conv_i-1], conv_dim, kernel_size=11, stride=1, padding=5))
                conv_list.append(nn.ELU())
            self.encoder_model = nn.Sequential(*conv_list)
            patch_dim = conv_dims[-1] + 3
        elif self.encoder == 'resnet18':
            self.conv_dims = 'default'
            self.encoder_num_layers = 4
            self.encoder_model = getattr(torchvision.models, 'resnet18')(pretrained=True)

            if in_dim != 3:
                self.encoder_model.conv1 = nn.Conv2d(in_dim, 64, (7,7), (2,2), (3,3))

            if remove_unused_layers:
                self.encoder_model.layer4 = nn.Identity()
                self.encoder_model.fc = nn.Identity()
            self.encode(None, torch.zeros((1,in_dim,224,224)))
            if verbose:
                print('using', self.encoder, 'as encoder backbone with feat_size', self.feat_size)
            patch_dim = self.feat_size + self.in_dim

        #! FEATURE DIMS
        ray_dim = self.harmonic_embedding.get_output_dim(6)
        depth_dim = self.harmonic_embedding.get_output_dim(1)

        #! T1: Vision
        intermediate_dim = 256
        t1_in =  ray_dim + depth_dim + patch_dim
        self.t1 = TransformerEncoder(t1_in, intermediate_dim)

        #! T2: Epipolar
        if self.use_r:
            t2_in =  2*ray_dim + depth_dim + intermediate_dim
        else:
            t2_in = 1*ray_dim + depth_dim + intermediate_dim
        self.t2 = TransformerEncoder(t2_in, intermediate_dim)
        self.t2_attn = nn.Linear(intermediate_dim, 1)

        #! T3: Color
        if self.use_r:
            t3_in = 2*ray_dim + intermediate_dim
        else:
            t3_in = 1*ray_dim + intermediate_dim
        self.t3 = TransformerEncoder(t3_in, intermediate_dim)
        self.t3_attn = nn.Linear(intermediate_dim, 1)

        if out_sigmoid:
            self.color_layer = nn.Sequential(nn.Linear(intermediate_dim, out_dim), nn.Sigmoid())
        else:
            self.color_layer = nn.Sequential(nn.Linear(intermediate_dim, out_dim))

        self.input_bbox = None
    
    def get_config(self):
        """
        Returns dictionary of config
        """
        config = {
            'model':'patch_nerf',
            'conv_dims':self.conv_dims, 
            'encoder':self.encoder,
        }
        return config

    def encode(
        self,
        input_cameras,
        input_images,
        input_bbox=None
    ):
        """
        Encode cameras and images
        """
        #! ENCODE IMAGES
        if input_images.shape[1] != self.in_dim:
            input_images = input_images.permute(0,3,1,2)
        self.input_images = input_images
        self.input_cameras = input_cameras
        self.input_bbox = input_bbox
        #' input images (B, 3, H, W)

        if self.encoder == 'lite':
            self.encoder_latent = self.encoder_model(input_images)
            self.feat_size = self.encoder_latent.shape[1]
        elif self.encoder == 'resnet18':
            x = self.encoder_model.conv1(input_images)
            x = self.encoder_model.bn1(x)
            x = self.encoder_model.relu(x)

            latents = [x]
            if self.encoder_num_layers > 1:
                x = self.encoder_model.maxpool(x)
                x = self.encoder_model.layer1(x)
                latents.append(x)
            if self.encoder_num_layers > 2:
                x = self.encoder_model.layer2(x)
                latents.append(x)
            if self.encoder_num_layers > 3:
                x = self.encoder_model.layer3(x)
                latents.append(x)
            if self.encoder_num_layers > 4:
                x = self.encoder_model.layer4(x)
                latents.append(x)

            latent_sz = latents[0].shape[-2:]
            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode='bilinear',
                    align_corners=True,
                )
            
            self.encoder_latent = torch.cat(latents, dim=1)
            self.feat_size = self.encoder_latent.shape[1]
        return self.input_images, self.encoder_latent

    def encode_plucker(self, ray_origins, ray_dirs):
        """
        ray to plucker w/ pos encoding
        """
        plucker = torch.cat((ray_dirs, torch.cross(ray_origins, ray_dirs, dim=-1)), dim=-1)
        plucker = self.harmonic_embedding(plucker)
        return plucker 

    def index(
        self,
        xyz_world,
        ray_origins, 
        ray_dirs,
        ray_depths,
        ray_xys,
        ray_bundle,
        input_images=None,
        encoder_latent=None,
    ):
        """
        get epipolar features
        """

        #! GET EPIPOLAR PTS
        #' xyz_world (1, N, PTS_PER_RAY, 3)

        xyz_ = xyz_world.reshape((xyz_world.shape[0],-1,3))
        if xyz_.shape[0] > 1:
            xyz_ = xyz_.reshape((1,-1,3))
        #' xyz_ (1, N*PTS_PER_RAY, 3) - reshape to use transform_points

        xyz_cam = self.input_cameras.transform_points_ndc(xyz_)
        #' xyz_cam (NUM_INPUT_CAM, N*PTS_PER_RAY, 3)

        xy_cam = xyz_cam[...,:2].unsqueeze(2)
        #' xy (NUM_INPUT_CAM, N*PTS_PER_RAY, 1, 2)

        if input_images is None:
            input_images = self.input_images
        if encoder_latent is None:
            encoder_latent = self.encoder_latent

        #! GET FEATURES
        features = F.grid_sample(
                encoder_latent, # (B, L, H, W)
                -xy_cam, # (B, N, 1, 2)
                align_corners=True,
                mode='bilinear',
                padding_mode='border',
            )
        #' features (NUM_INPUT_CAM, FEATURE_DIM, N*PTS_PER_RAY, 1)
        
        features = features[...,0].permute(0,2,1)
        #' features (NUM_INPUT_CAM, N*PTS_PER_RAY, FEATURE_DIM)

        if self.input_bbox is not None:
            min_x = -self.input_bbox[:,1].unsqueeze(1).unsqueeze(1)
            max_x = -self.input_bbox[:,3].unsqueeze(1).unsqueeze(1)
            min_y = -self.input_bbox[:,0].unsqueeze(1).unsqueeze(1)
            max_y = -self.input_bbox[:,2].unsqueeze(1).unsqueeze(1)
            ep_bool = torch.ones_like(xy_cam[...,0])
            ep_sum = ep_bool.sum()
            # print(xy_cam[:,:5,...])
            # print(min_y)
            # print(min_y, min_x, max_y, max_x)
            ep_bool[xy_cam[...,0] > min_y] = 0
            # print('invalid:', (ep_sum - ep_bool.sum()).item(), 'out of:', ep_sum.item())
            # ep_bool = torch.ones_like(xy_cam[...,0])
            ep_bool[xy_cam[...,0] < max_y] = 0
            # print('invalid:', (ep_sum - ep_bool.sum()).item(), 'out of:', ep_sum.item())
            # ep_bool = torch.ones_like(xy_cam[...,0])
            ep_bool[xy_cam[...,1] > min_x] = 0
            # print('invalid:', (ep_sum - ep_bool.sum()).item(), 'out of:', ep_sum.item())
            # ep_bool = torch.ones_like(xy_cam[...,0])
            ep_bool[xy_cam[...,1] < max_x] = 0
            # print('invalid:', (ep_sum - ep_bool.sum()).item(), 'out of:', ep_sum.item())
            ep_bool = torch.ones_like(xy_cam[...,0])
            features = features * ep_bool
            # exit(0)

        #! GET RGB
        rgb_feats = F.grid_sample(
                input_images, # (B, L, H, W)
                -xy_cam, # (B, N, 1, 2)
                align_corners=True,
                mode='bilinear',
                padding_mode='border',
            )
        rgb_feats = rgb_feats[...,0].permute(0,2,1)
        #' rgb_feats (NUM_INPUT_CAM, N*PTS_PER_RAY, IN_DIM)

        if len(xyz_world.shape) > 3:
            # print(features.shape, xyz_world.shape)
            features = features.reshape((len(self.input_cameras),xyz_world.shape[1],xyz_world.shape[2],self.feat_size))
            rgb_feats = rgb_feats.reshape((len(self.input_cameras),xyz_world.shape[1],xyz_world.shape[2],self.in_dim))
        else:
            # print(features.shape, xyz_world.shape)
            features = features.reshape((len(self.input_cameras),xyz_world.shape[0],xyz_world.shape[1],self.feat_size))
            rgb_feats = rgb_feats.reshape((len(self.input_cameras),xyz_world.shape[0],xyz_world.shape[1],self.in_dim))
        #' features (NUM_INPUT_CAM, N, PTS_PER_RAY, FEATURE_DIM)
        #' rgb_feats (NUM_INPUT_CAM, N, PTS_PER_RAY, 3)

        #! CONCAT FEATURES
        features = torch.cat((features, rgb_feats), dim=-1)
        #' features (NUM_INPUT_CAM, N, PTS_PER_RAY, FEATURE_DIM + 3)
        

        #! GET REFERNCE PLUCKER
        origins_cam = self.input_cameras.get_camera_center()
        origins_cam = repeat(origins_cam, 'n c -> n b d c', b=xyz_world.shape[-3], d=xyz_world.shape[-2])
        #' origins_cam (NUM_INPUT_CAM, N, 1, 3)

        #! R
        if len(xyz_world.shape) > 3:
            input_dirs = (xyz_world - origins_cam)
        else:
            input_dirs = (rearrange(xyz_world, 'b d c -> () b d c') - origins_cam)
        vis_depths = torch.linalg.norm(input_dirs, dim=-1, keepdim=True)[...,:,:]
        input_dirs = torch.nn.functional.normalize(input_dirs, dim=-1)
        #' input_dirs (NUM_INPUT_CAM, N, D, 3)
        
        reference_plucker = self.encode_plucker(origins_cam, input_dirs)
        #' reference_plucker (NUM_INPUT_CAM, N, D, PLUCKER_DIM)

        #! GET DEPTH
        depths = self.harmonic_embedding(ray_depths.unsqueeze(-1))
        if len(depths.shape) == 3:
            depths = depths.unsqueeze(0)
        #' depths (1, N, PTS_PER_RAY, DEPTH_DIM)
        return reference_plucker, depths, features
    
    def get_coarse_rgb(self, features, t2_w, t3_w):
        """
        Use linear combination to get corase rgb
        """
        ref_rgb = features[...,-3:]
        #' ref_rgb (NUM_INPUT_CAM, N, P, 3)

        neighbor_rgb = (ref_rgb * t2_w).sum(-2)
        #' neighbor_rgb (NUM_INPUT_CAM, N, 3)

        coarse_rgb = (neighbor_rgb * t3_w).sum(0)
        #' coarse_rgb (N, 3)
  
        coarse_rgb = torch.clip(coarse_rgb, 0, 1)
        
        return coarse_rgb
  
    def forward(
        self, 
        ray_bundle: RayBundle,
        return_intermediates = False,
        **kwargs,
    ):
        """
        - convert to rel cameras 
        - ray -> plucker
        - ray -> feat
        - aggregate features
        - feed to transformer
        - return colors
        """

        #! GET FEATURES
        if 'input_cameras' in kwargs and kwargs['input_cameras'] is not None:
            if 'input_bbox' in kwargs:
                input_images, encoder_latent = self.encode(kwargs['input_cameras'], kwargs['input_rgb'], input_bbox=kwargs['input_bbox'])
            else:
                input_images, encoder_latent = self.encode(kwargs['input_cameras'], kwargs['input_rgb'])
        
        #! EXTRACT RAY
        xyz_world = ray_bundle_to_ray_points(ray_bundle)
        #' ray_dirs (B, N, PTS_PER_RAY, 3)

        ray_origins = ray_bundle.origins
        #' ray_dirs (B, N, 3)
        

        ray_dirs = torch.nn.functional.normalize(ray_bundle.directions, dim=-1)
        #' ray_dirs (B, N, 3)

        
        query_plucker = self.encode_plucker(ray_origins, ray_dirs).unsqueeze(-2)
        if xyz_world.shape[0] > 1:
            query_plucker = query_plucker.unsqueeze(0)
        #' query_plucker (1, N, 1, 78)

        #! SAMPLE EPIPOLAR FEATURES
        if 'input_cameras' in kwargs:
            reference_plucker, depths, features = self.index(xyz_world, ray_origins, ray_bundle.directions, ray_bundle.lengths, ray_bundle.xys, ray_bundle, input_images, encoder_latent)
        else:
            reference_plucker, depths, features = self.index(xyz_world, ray_origins, ray_bundle.directions, ray_bundle.lengths, ray_bundle.xys, ray_bundle)
        #' reference_plucker (NUM_INPUT_CAM, N, D, PLUCKER_DIM)
        #' depths (1, N, PTS_PER_RAY, DEPTH_DIM)
        #' features (NUM_INPUT_CAM, N, PTS_PER_RAY, FEATURE_DIM + 3)
        NC, N = reference_plucker.shape[0], reference_plucker.shape[1]
        D = depths.shape[2]

        #! T1
        # INPUT (S, B, D) - (NUM_INPUT_CAMERAS, N*PTS_PER_RAY, D)
        t1_reference_plucker = rearrange(reference_plucker, 'nc n d f -> nc (n d) f')
        t1_depths = rearrange(depths.expand(NC,-1,-1,-1), 'nc n d f -> nc (n d) f')
        t1_features = rearrange(features, 'nc n d f -> nc (n d) f')
        f1 = self.t1(torch.cat((t1_reference_plucker, t1_depths, t1_features), dim=-1))
        f1 = rearrange(f1, 'nc (n d) f -> nc n d f', n=N, d=D)
        #' f1 (NUM_INPUT_CAMERAS, N, D, F)
        

        #! T2
        # INPUT (S, B, D) - (PTS_PER_RAY, N*NUM_INPUT_CAMERAS , D)
        t2_query_plucker = rearrange(query_plucker.expand(NC,-1,D,-1), 'nc n d f -> d (nc n) f')
        t2_reference_plucker = rearrange(reference_plucker, 'nc n d f -> d (nc n) f')
        t2_depths = rearrange(depths.expand(NC,-1,-1,-1), 'nc n d f -> d (nc n) f')
        t2_features = rearrange(f1, 'nc n d f -> d (nc n) f')
        if self.use_r:
            f2 = self.t2(torch.cat((t2_query_plucker, t2_reference_plucker, t2_depths, t2_features), dim=-1))
        else:
            f2 = self.t2(torch.cat((t2_query_plucker, t2_depths, t2_features), dim=-1))
        f2 = rearrange(f2, 'd (nc n) f -> nc n d f', n=N, d=D)
        #' f2 (NUM_INPUT_CAMERAS, N, D, F)

        t2 = self.t2_attn(f2)
        t2_w = nn.functional.softmax(t2, dim=-2)
        #' t2_w (NUM_INPUT_CAMERAS, N, D, 1)

        f2 = (f2 * t2_w).sum(dim=-2)
        #' f2 (NUM_INPUT_CAMERAS, N, D)
        

        #! T3
        # INPUT (S, B, D) - (NUM_INPUT_CAMERAS, N, D)
        t3_query_plucker = query_plucker.expand(NC,-1,-1,-1)[...,0,:]
        t3_reference_plucker = reference_plucker[...,(D//2),:]
        if self.use_r:
            f3 = self.t3(torch.cat((t3_query_plucker, t3_reference_plucker, f2), dim=-1))
        else:
            f3 = self.t3(torch.cat((t3_query_plucker, f2), dim=-1))
        t3_w = nn.functional.softmax(self.t3_attn(f3), dim=0)
        #' t3_w (NUM_INPUT_CAMERAS, N, D)

        f3 = (f3 * t3_w).sum(dim=0)
        #' f3 (N, D)


        rgb = self.color_layer(f3)
        #' rgb (N, 3)

        if self.return_features:
            return rgb, f3, 0

        #! COARSE RGB
        coarse_rgb = self.get_coarse_rgb(features, t2_w, t3_w)

        if len(ray_origins.shape) == 3:
            rgb = rgb[None, ...]
            coarse_rgb = coarse_rgb[None, ...]
        
        if return_intermediates:
            return rgb, coarse_rgb, t2_w, t3_w
            
        return rgb, coarse_rgb, 0
    
    def batched_forward(
        self, 
        ray_bundle: RayBundle,
        n_batches: int = 32,
        return_intermediates = False,
        **kwargs,        
    ):
        """
        """
        if 'input_cameras' in kwargs and kwargs['input_cameras'] is not None:
            input_images, encoder_latent = self.encode(kwargs['input_cameras'], kwargs['input_rgb'])

        # Parse out shapes needed for tensor reshaping in this function.
        n_pts_per_ray = ray_bundle.lengths.shape[-1]  
        # spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]
        spatial_size = [*ray_bundle.origins.shape[:-1]]
        spatial_size_t2 = [len(self.input_images), *ray_bundle.origins.shape[1:-1], n_pts_per_ray]
        spatial_size_t3 = [len(self.input_images), *ray_bundle.origins.shape[1:-1]]

        # Split the rays to `n_batches` batches.
        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)

        # For each batch, execute the standard forward pass.
        batch_outputs = [
            self.forward(
                RayBundle(
                    origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                    directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                    lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                    xys=None,
                ),
                return_intermediates=return_intermediates
            ) for batch_idx in batches
        ]
                
        
        rays_densities, rays_colors = [
            torch.cat(
                [batch_output[output_i] for batch_output in batch_outputs], dim=0
            ).view(*spatial_size, -1) for output_i in (0, 1)
        ]
        # return rays_densities, rays_colors

        if not return_intermediates:
            return rays_densities, rays_colors, 0

        else:
            t2_w = torch.cat([batch_output[2] for batch_output in batch_outputs], dim=0).view(*spatial_size_t2, -1)
            #' t2_w (NUM_INPUT_CAMERAS, H, W, D, 1)

            t3_w = torch.cat([batch_output[3] for batch_output in batch_outputs], dim=0).view(*spatial_size_t3, -1)
            #' t3_w (NUM_INPUT_CAMERAS, H, W, 1)
            
            # return rays_densities, rays_colors
            return rays_densities, rays_colors, t2_w, t3_w 
