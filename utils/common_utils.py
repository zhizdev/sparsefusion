'''
A collection of common utilities
'''
import torch
import lpips
import skimage.metrics


def normalize(x):
    '''
    Normalize [0, 1] to [-1, 1]
    '''
    return torch.clip(x*2 - 1.0, -1.0, 1.0)

def unnormalize(x):
    '''
    Unnormalize [-1, 1] to [0, 1]
    '''
    return torch.clip((x + 1.0) / 2.0, 0.0, 1.0)

def split_list(a, n):
    '''
    Split list into n parts

    Args:
        a (list): list
        n (int): number of parts
    
    Returns:
        a_split (list[list]): nested list of a split into n parts
    '''
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


def get_lpips_fn():
    '''
    Return LPIPS function
    '''
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    return loss_fn_vgg


def get_metrics(pred, gt, use_lpips=False, loss_fn_vgg=None, device=None):
    '''
    Compute image metrics 

    Args:
        pred (np array): (H, W, 3)
        gt (np array): (H, W, 3)
    '''
    ssim = skimage.metrics.structural_similarity(pred, gt, channel_axis = -1, data_range=1)
    psnr = skimage.metrics.peak_signal_noise_ratio(pred, gt, data_range=1)
    if use_lpips:
        if loss_fn_vgg is None:
            loss_fn_vgg = lpips.LPIPS(net='vgg')
        gt_ = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0)*2 - 1.0
        pred_ = torch.from_numpy(pred).permute(2,0,1).unsqueeze(0)*2 - 1.0
        if device is not None:
            lp = loss_fn_vgg(gt_.to(device), pred_.to(device)).detach().cpu().numpy().item()
        else:
            lp = loss_fn_vgg(gt_, pred_).detach().numpy().item()
        return ssim, psnr, lp
    return ssim, psnr


#@ FROM PyTorch3D
class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions: int = 6,
        omega_0: float = 1.0,
        logspace: bool = True,
        append_input: bool = True,
    ) -> None:
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        (i.e. vector along the last dimension) in `x`
        into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]:
            ```
            [
                sin(f_1*x[..., i]),
                sin(f_2*x[..., i]),
                ...
                sin(f_N * x[..., i]),
                cos(f_1*x[..., i]),
                cos(f_2*x[..., i]),
                ...
                cos(f_N * x[..., i]),
                x[..., i],              # only present if append_input is True.
            ]
            ```
        where N corresponds to `n_harmonic_functions-1`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.
        If `logspace==True`, the frequencies `[f_1, ..., f_N]` are
        powers of 2:
            `f_1, ..., f_N = 2**torch.arange(n_harmonic_functions)`
        If `logspace==False`, frequencies are linearly spaced between
        `1.0` and `2**(n_harmonic_functions-1)`:
            `f_1, ..., f_N = torch.linspace(
                1.0, 2**(n_harmonic_functions-1), n_harmonic_functions
            )`
        Note that `x` is also premultiplied by the base frequency `omega_0`
        before evaluating the harmonic functions.
        Args:
            n_harmonic_functions: int, number of harmonic
                features
            omega_0: float, base frequency
            logspace: bool, Whether to space the frequencies in
                logspace or linear space
            append_input: bool, whether to concat the original
                input to the harmonic embedding. If true the
                output is of the form (x, embed.sin(), embed.cos()
        """
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", frequencies * omega_0, persistent=False)
        self.append_input = append_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., (n_harmonic_functions * 2 + int(append_input)) * dim]
        """
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)
        embed = torch.cat(
            (embed.sin(), embed.cos(), x)
            if self.append_input
            else (embed.sin(), embed.cos()),
            dim=-1,
        )
        return embed

    @staticmethod
    def get_output_dim_static(
        input_dims: int,
        n_harmonic_functions: int,
        append_input: bool,
    ) -> int:
        """
        Utility to help predict the shape of the output of `forward`.
        Args:
            input_dims: length of the last dimension of the input tensor
            n_harmonic_functions: number of embedding frequencies
            append_input: whether or not to concat the original
                input to the harmonic embedding
        Returns:
            int: the length of the last dimension of the output tensor
        """
        return input_dims * (2 * n_harmonic_functions + int(append_input))

    def get_output_dim(self, input_dims: int = 3) -> int:
        """
        Same as above. The default for input_dims is 3 for 3D applications
        which use harmonic embedding for positional encoding,
        so the input might be xyz.
        """
        return self.get_output_dim_static(
            input_dims, len(self._frequencies), self.append_input
        )

    
#@ From PyTorch3D
def huber(x, y, scaling=0.1):
    """
    A helper function for evaluating the smooth L1 (huber) loss
    between the rendered silhouettes and colors.
    """
    diff_sq = (x - y) ** 2
    loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss


#@ From PyTorch3D
def sample_images_at_mc_locs(target_images, sampled_rays_xy):
    """
    Given a set of Monte Carlo pixel locations `sampled_rays_xy`,
    this method samples the tensor `target_images` at the
    respective 2D locations.
    
    This function is used in order to extract the colors from
    ground truth images that correspond to the colors
    rendered using `MonteCarloRaysampler`.
    """
    ba = target_images.shape[0]
    dim = target_images.shape[-1]
    spatial_size = sampled_rays_xy.shape[1:-1]
    # In order to sample target_images, we utilize
    # the grid_sample function which implements a
    # bilinear image sampler.
    # Note that we have to invert the sign of the 
    # sampled ray positions to convert the NDC xy locations
    # of the MonteCarloRaysampler to the coordinate
    # convention of grid_sample.

    if target_images.shape[2] != target_images.shape[3]:
        target_images = target_images.permute(0, 3, 1, 2)
    elif target_images.shape[2] == target_images.shape[3]:
        dim = target_images.shape[1]

    images_sampled = torch.nn.functional.grid_sample(
        target_images, 
        -sampled_rays_xy.view(ba, -1, 1, 2),  # note the sign inversion
        align_corners=True
    )
    return images_sampled.permute(0, 2, 3, 1).view(
        ba, *spatial_size, dim
    )
