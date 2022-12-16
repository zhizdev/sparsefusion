'''
Custom PyTorch3D Raymarcher for EFT
#@ Modified from https://github.com/facebookresearch/pytorch3d
'''

from typing import Optional, Tuple, Union
import torch

from pytorch3d.renderer import EmissionAbsorptionRaymarcher
from pytorch3d.renderer.implicit.raymarching import (
    _check_density_bounds,
    _check_raymarcher_inputs,
    _shifted_cumprod,
)

class LightFieldRaymarcher(torch.nn.Module):
    """
    A nominal ray marcher that returns LightField features without any raymarching
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        **kwargs
    ) -> Union[None, torch.Tensor]:
        """
        """
        return torch.cat((rays_densities, rays_features), dim=-1)