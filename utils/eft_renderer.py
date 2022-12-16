'''
Custom PyTorch3D Implicit Renderer
#@ Modified from https://github.com/facebookresearch/pytorch3d
'''

from typing import Callable, Tuple
from einops import rearrange, repeat, reduce
import torch

from pytorch3d.ops.utils import eyes
from pytorch3d.structures import Volumes
from pytorch3d.transforms import Transform3d
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit.raysampling import RayBundle
from pytorch3d.renderer.implicit.utils import _validate_ray_bundle_variables, ray_bundle_variables_to_ray_points, ray_bundle_to_ray_points

#@ MODIFIED FROM PYTORCH3D
class CustomImplicitRenderer(torch.nn.Module):
    """
    A class for rendering a batch of implicit surfaces. The class should
    be initialized with a raysampler and raymarcher class which both have
    to be a `Callable`.
    VOLUMETRIC_FUNCTION
    The `forward` function of the renderer accepts as input the rendering cameras
    as well as the `volumetric_function` `Callable`, which defines a field of opacity
    and feature vectors over the 3D domain of the scene.
    A standard `volumetric_function` has the following signature:
    ```
    def volumetric_function(
        ray_bundle: RayBundle,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]
    ```
    With the following arguments:
        `ray_bundle`: A RayBundle object containing the following variables:
            `origins`: A tensor of shape `(minibatch, ..., 3)` denoting
                the origins of the rendering rays.
            `directions`: A tensor of shape `(minibatch, ..., 3)`
                containing the direction vectors of rendering rays.
            `lengths`: A tensor of shape
                `(minibatch, ..., num_points_per_ray)`containing the
                lengths at which the ray points are sampled.
            `xys`: A tensor of shape
                `(minibatch, ..., 2)` containing the
                xy locations of each ray's pixel in the screen space.
    Calling `volumetric_function` then returns the following:
        `rays_densities`: A tensor of shape
            `(minibatch, ..., num_points_per_ray, opacity_dim)` containing
            the an opacity vector for each ray point.
        `rays_features`: A tensor of shape
            `(minibatch, ..., num_points_per_ray, feature_dim)` containing
            the an feature vector for each ray point.
    Note that, in order to increase flexibility of the API, we allow multiple
    other arguments to enter the volumetric function via additional
    (optional) keyword arguments `**kwargs`.
    A typical use-case is passing a `CamerasBase` object as an additional
    keyword argument, which can allow the volumetric function to adjust its
    outputs based on the directions of the projection rays.
    Example:
        A simple volumetric function of a 0-centered
        RGB sphere with a unit diameter is defined as follows:
        ```
        def volumetric_function(
            ray_bundle: RayBundle,
            **kwargs,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            # first convert the ray origins, directions and lengths
            # to 3D ray point locations in world coords
            rays_points_world = ray_bundle_to_ray_points(ray_bundle)
            # set the densities as an inverse sigmoid of the
            # ray point distance from the sphere centroid
            rays_densities = torch.sigmoid(
                -100.0 * rays_points_world.norm(dim=-1, keepdim=True)
            )
            # set the ray features to RGB colors proportional
            # to the 3D location of the projection of ray points
            # on the sphere surface
            rays_features = torch.nn.functional.normalize(
                rays_points_world, dim=-1
            ) * 0.5 + 0.5
            return rays_densities, rays_features
        ```
    """

    def __init__(self, raysampler: Callable, raymarcher: Callable, reg=None) -> None:
        """
        Args:
            raysampler: A `Callable` that takes as input scene cameras
                (an instance of `CamerasBase`) and returns a `RayBundle` that 
                describes the rays emitted from the cameras.
            raymarcher: A `Callable` that receives the response of the
                `volumetric_function` (an input to `self.forward`) evaluated
                along the sampled rays, and renders the rays with a
                ray-marching algorithm.
        """
        super().__init__()

        if not callable(raysampler):
            raise ValueError('"raysampler" has to be a "Callable" object.')
        if not callable(raymarcher):
            raise ValueError('"raymarcher" has to be a "Callable" object.')

        self.raysampler = raysampler
        self.raymarcher = raymarcher
        self.reg = reg

    def forward(
        self, cameras: CamerasBase, volumetric_function: Callable, **kwargs
    ) -> Tuple[torch.Tensor, RayBundle]:
        """
        Render a batch of images using a volumetric function
        represented as a callable (e.g. a Pytorch module).
        Args:
            cameras: A batch of cameras that render the scene. A `self.raysampler`
                takes the cameras as input and samples rays that pass through the
                domain of the volumetric function.
            volumetric_function: A `Callable` that accepts the parametrizations
                of the rendering rays and returns the densities and features
                at the respective 3D of the rendering rays. Please refer to
                the main class documentation for details.
        Returns:
            images: A tensor of shape `(minibatch, ..., feature_dim + opacity_dim)`
                containing the result of the rendering.
            ray_bundle: A `RayBundle` containing the parametrizations of the
                sampled rendering rays.
        """

        if not callable(volumetric_function):
            raise ValueError('"volumetric_function" has to be a "Callable" object.')

        # first call the ray sampler that returns the RayBundle parametrizing
        # the rendering rays.
        ray_bundle = self.raysampler(
            cameras=cameras, volumetric_function=volumetric_function, **kwargs
        )
        # ray_bundle.origins - minibatch x ... x 3
        # ray_bundle.directions - minibatch x ... x 3
        # ray_bundle.lengths - minibatch x ... x n_pts_per_ray
        # ray_bundle.xys - minibatch x ... x 2

        # given sampled rays, call the volumetric function that
        # evaluates the densities and features at the locations of the
        # ray points
        if self.reg is not None:
            rays_densities, rays_features, reg_term = volumetric_function(
                ray_bundle=ray_bundle, cameras=cameras, **kwargs
            )
        else:
            rays_densities, rays_features, _ = volumetric_function(
                ray_bundle=ray_bundle, cameras=cameras, **kwargs
            )
        # ray_densities - minibatch x ... x n_pts_per_ray x density_dim
        # ray_features - minibatch x ... x n_pts_per_ray x feature_dim

        # finally, march along the sampled rays to obtain the renders
        images = self.raymarcher(
            rays_densities=rays_densities,
            rays_features=rays_features,
            ray_bundle=ray_bundle,
            **kwargs
        )
        # images - minibatch x ... x (feature_dim + opacity_dim)

        if self.reg is not None:
            return images, ray_bundle, reg_term
        else:
            return images, ray_bundle, 0