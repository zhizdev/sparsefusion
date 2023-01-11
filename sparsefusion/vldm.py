'''
Class for View-conditioned Diffusion
#@ Based upon https://github.com/lucidrains/imagen-pytorch
'''

import math
import copy
from typing import List, Union
from tqdm import tqdm
from functools import partial, wraps
from contextlib import contextmanager, nullcontext
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.cuda.amp import autocast
from torch.special import expm1
import torchvision.transforms as T

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from external.einops_exts import rearrange_many, repeat_many, check_shape
from external.imagen_pytorch import (
    EinopsToAndFrom,
    Identity,
    GaussianDiffusionContinuousTimes,
    Unet,
    exists,
    identity,
    first,
    maybe,
    once,
    default,
    cast_tuple,
    cast_uint8_images_to_float,
    module_device,
    zero_init_,
    eval_decorator,
    pad_tuple_to_length,
    log,
    l2norm,
    right_pad_dims_to,
    masked_mean,
    resize_image_to,
    normalize_neg_one_to_one,
    unnormalize_zero_to_one,
    prob_mask_like,   
)


class DDPM(nn.Module):
    def __init__(
        self,
        unets,
        *,
        image_sizes,                                # for cascading ddpm, image size at each stage
        conditional_encoder,
        conditional_embed_dim = 1024,
        channels = 3,
        timesteps = 1000,
        cond_drop_prob = 0.1,
        loss_type = 'l2',
        noise_schedules = 'cosine',
        pred_objectives = 'noise',
        random_crop_sizes = None,
        lowres_noise_schedule = 'linear',
        lowres_sample_noise_level = 0.2,            # in the paper, they present a new trick where they noise the lowres conditioning image, and at sample time, fix it to a certain level (0.1 or 0.3) - the unets are also made to be conditioned on this noise level
        per_sample_random_aug_noise_level = False,  # unclear when conditioning on augmentation noise level, whether each batch element receives a random aug noise value - turning off due to @marunine's find
        conditional = True,
        auto_normalize_img = False,                  # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
        p2_loss_weight_gamma = 0.5,                 # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time
        p2_loss_weight_k = 1,
        dynamic_thresholding = True,
        dynamic_thresholding_percentile = 0.95,     # unsure what this was based on perusal of paper
        only_train_unet_number = None,
        clip_output = True,
        clip_value = 1.0,
    ):
        super().__init__()

        self.timesteps = timesteps

        # loss

        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        self.loss_type = loss_type
        self.loss_fn = loss_fn

        # conditioning hparams

        self.conditional = conditional
        self.unconditional = not conditional

        # channels

        self.channels = channels

        # automatically take care of ensuring that first unet is unconditional
        # while the rest of the unets are conditioned on the low resolution image produced by previous unet

        unets = cast_tuple(unets)
        num_unets = len(unets)

        # determine noise schedules per unet

        timesteps = cast_tuple(timesteps, num_unets)

        # make sure noise schedule defaults to 'cosine', 'cosine', and then 'linear' for rest of super-resoluting unets

        noise_schedules = cast_tuple(noise_schedules)
        noise_schedules = pad_tuple_to_length(noise_schedules, 2, 'cosine')
        noise_schedules = pad_tuple_to_length(noise_schedules, num_unets, 'linear')

        # construct noise schedulers

        noise_scheduler_klass = GaussianDiffusionContinuousTimes
        self.noise_schedulers = nn.ModuleList([])

        for timestep, noise_schedule in zip(timesteps, noise_schedules):
            noise_scheduler = noise_scheduler_klass(noise_schedule = noise_schedule, timesteps = timestep)
            self.noise_schedulers.append(noise_scheduler)

        # randomly cropping for upsampler training

        self.random_crop_sizes = cast_tuple(random_crop_sizes, num_unets)
        assert not exists(first(self.random_crop_sizes)), 'you should not need to randomly crop image during training for base unet, only for upsamplers - so pass in `random_crop_sizes = (None, 128, 256)` as example'

        # lowres augmentation noise schedule

        self.lowres_noise_schedule = GaussianDiffusionContinuousTimes(noise_schedule = lowres_noise_schedule)

        # ddpm objectives - predicting noise by default

        self.pred_objectives = cast_tuple(pred_objectives, num_unets)

        # get text encoder

        self.conditional_embed_dim = conditional_embed_dim

        #! REPLACE ENCODER
        # self.encode_text = partial(t5_encode_text, name = conditional_encoder_name)
        self.encode_conditional = conditional_encoder

        # construct unets

        self.unets = nn.ModuleList([])

        self.unet_being_trained_index = -1 # keeps track of which unet is being trained at the moment
        self.only_train_unet_number = only_train_unet_number

        for ind, one_unet in enumerate(unets):
            assert isinstance(one_unet, (Unet))
            is_first = ind == 0

            one_unet = one_unet.cast_model_parameters(
                lowres_cond = not is_first,
                cond_on_z = self.conditional,
                conditional_embed_dim = self.conditional_embed_dim if self.conditional else None,
                channels = self.channels,
                channels_out = self.channels
            )

            self.unets.append(one_unet)

        # unet image sizes

        image_sizes = cast_tuple(image_sizes)
        self.image_sizes = image_sizes

        assert num_unets == len(image_sizes), f'you did not supply the correct number of u-nets ({len(unets)}) for resolutions {image_sizes}'

        self.sample_channels = cast_tuple(self.channels, num_unets)

        # determine whether we are training on images or video

        # is_video = any([isinstance(unet, Unet) for unet in self.unets])
        is_video = False
        self.is_video = is_video

        self.right_pad_dims_to_datatype = partial(rearrange, pattern = ('b -> b 1 1 1' if not is_video else 'b -> b 1 1 1 1'))
        self.resize_to = resize_image_to

        # cascading ddpm related stuff

        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (False, *((True,) * (num_unets - 1))), 'the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True'

        self.lowres_sample_noise_level = lowres_sample_noise_level
        self.per_sample_random_aug_noise_level = per_sample_random_aug_noise_level

        # classifier free guidance

        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.

        # normalize and unnormalize image functions

        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity
        self.input_image_range = (0. if auto_normalize_img else -1., 1.)

        # dynamic thresholding

        self.dynamic_thresholding = cast_tuple(dynamic_thresholding, num_unets)
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile
        self.clip_output = clip_output
        self.clip_value = clip_value

        # p2 loss weight

        self.p2_loss_weight_k = p2_loss_weight_k
        self.p2_loss_weight_gamma = cast_tuple(p2_loss_weight_gamma, num_unets)

        assert all([(gamma_value <= 2) for gamma_value in self.p2_loss_weight_gamma]), 'in paper, they noticed any gamma greater than 2 is harmful'

        # one temp parameter for keeping track of device

        self.register_buffer('_temp', torch.tensor([0.]), persistent = False)

        # default to device of unets passed in

        self.to(next(self.unets.parameters()).device)

    @property
    def device(self):
        return self._temp.device

    def get_unet(self, unet_number):
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1

        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.unets]
            delattr(self, 'unets')
            self.unets = unets_list

        if index != self.unet_being_trained_index:
            for unet_index, unet in enumerate(self.unets):
                unet.to(self.device if unet_index == index else 'cpu')

        self.unet_being_trained_index = index
        return self.unets[index]

    def reset_unets_all_one_device(self, device = None):
        device = default(device, self.device)
        self.unets = nn.ModuleList([*self.unets])
        self.unets.to(device)

        self.unet_being_trained_index = -1

    @contextmanager
    def one_unet_in_gpu(self, unet_number = None, unet = None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.unets[unet_number - 1]

        devices = [module_device(unet) for unet in self.unets]
        self.unets.cpu()
        unet.to(self.device)

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)

    # overriding state dict functions

    def state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().load_state_dict(*args, **kwargs)

    # gaussian diffusion methods

    def p_mean_variance(
        self,
        unet,
        x,
        t,
        *,
        noise_scheduler,
        conditional_embeds = None,
        conditional_mask = None,
        cond_images = None,
        lowres_cond_img = None,
        lowres_noise_times = None,
        cond_scale = 1.,
        model_output = None,
        t_next = None,
        pred_objective = 'noise',
        dynamic_threshold = True
    ):
        assert not (cond_scale != 1. and not self.can_classifier_guidance), 'imagen was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        pred = default(model_output, lambda: unet.forward_with_cond_scale(x, noise_scheduler.get_condition(t), conditional_embeds = conditional_embeds, conditional_mask = conditional_mask, cond_images = cond_images, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_times = self.lowres_noise_schedule.get_condition(lowres_noise_times)))

        if pred_objective == 'noise':            
            x_start = noise_scheduler.predict_start_from_noise(x, t = t, noise = pred)
        elif pred_objective == 'x_start':
            x_start = pred
        else:
            raise ValueError(f'unknown objective {pred_objective}')

        if self.clip_output:
            if dynamic_threshold:
                # following pseudocode in appendix
                # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element
                s = torch.quantile(
                    rearrange(x_start, 'b ... -> b (...)').abs(),
                    self.dynamic_thresholding_percentile,
                    dim = -1
                )
                
                s.clamp_(min = 1.)
                s = right_pad_dims_to(x_start, s)
                x_start = x_start.clamp(-s, s) / s
            else:
                x_start.clamp_(-self.clip_value, self.clip_value)

        return noise_scheduler.q_posterior(x_start = x_start, x_t = x, t = t, t_next = t_next)

    @torch.no_grad()
    def p_sample(
        self,
        unet,
        x,
        t,
        *,
        noise_scheduler,
        t_next = None,
        conditional_embeds = None,
        conditional_mask = None,
        cond_images = None,
        cond_scale = 1.,
        lowres_cond_img = None,
        lowres_noise_times = None,
        pred_objective = 'noise',
        dynamic_threshold = True
    ):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(unet, x = x, t = t, t_next = t_next, noise_scheduler = noise_scheduler, conditional_embeds = conditional_embeds, conditional_mask = conditional_mask, cond_images = cond_images, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_times = lowres_noise_times, pred_objective = pred_objective, dynamic_threshold = dynamic_threshold)
        noise = torch.randn_like(x)
        # no noise when t == 0
        is_last_sampling_timestep = (t_next == 0) if isinstance(noise_scheduler, GaussianDiffusionContinuousTimes) else (t == 0)
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        unet,
        shape,
        *,
        noise_scheduler,
        lowres_cond_img = None,
        lowres_noise_times = None,
        conditional_embeds = None,
        conditional_mask = None,
        cond_images = None,
        inpaint_images = None,
        inpaint_masks = None,
        inpaint_resample_times = 5,
        cond_scale = 1,
        pred_objective = 'noise',
        dynamic_threshold = True,
        use_tqdm = True
    ):
        device = self.device

        batch = shape[0]
        img = torch.randn(shape, device = device)

        # prepare inpainting

        has_inpainting = exists(inpaint_images) and exists(inpaint_masks)
        resample_times = inpaint_resample_times if has_inpainting else 1

        if has_inpainting:
            inpaint_images = self.normalize_img(inpaint_images)
            inpaint_images = self.resize_to(inpaint_images, shape[-1])
            inpaint_masks = self.resize_to(rearrange(inpaint_masks, 'b ... -> b 1 ...').float(), shape[-1]).bool()

        timesteps = noise_scheduler.get_sampling_timesteps(batch, device = device)

        for times, times_next in tqdm(timesteps, desc = 'sampling loop time step', total = len(timesteps), disable = not use_tqdm):
            is_last_timestep = times_next == 0

            for r in reversed(range(resample_times)):
                is_last_resample_step = r == 0

                if has_inpainting:
                    noised_inpaint_images, _ = noise_scheduler.q_sample(inpaint_images, t = times)
                    img = img * ~inpaint_masks + noised_inpaint_images * inpaint_masks

                img = self.p_sample(
                    unet,
                    img,
                    times,
                    t_next = times_next,
                    conditional_embeds = conditional_embeds,
                    conditional_mask = conditional_mask,
                    cond_images = cond_images,
                    cond_scale = cond_scale,
                    lowres_cond_img = lowres_cond_img,
                    lowres_noise_times = lowres_noise_times,
                    noise_scheduler = noise_scheduler,
                    pred_objective = pred_objective,
                    dynamic_threshold = dynamic_threshold
                )

                if has_inpainting and not (is_last_resample_step or torch.all(is_last_timestep)):
                    renoised_img = noise_scheduler.q_sample_from_to(img, times_next, times)

                    img = torch.where(
                        self.right_pad_dims_to_datatype(is_last_timestep),
                        img,
                        renoised_img
                    )

        if self.clip_output:
            img.clamp_(-self.clip_value, self.clip_value)

        # final inpainting

        if has_inpainting:
            img = img * ~inpaint_masks + inpaint_images * inpaint_masks

        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img

    @torch.no_grad()
    def sample(
        self,
        conditional_input = None,
        conditional_masks = None,
        conditional_embeds = None,
        video_frames = None,
        cond_images = None,
        inpaint_images = None,
        inpaint_masks = None,
        inpaint_resample_times = 5,
        batch_size = 1,
        cond_scale = 1.,
        lowres_sample_noise_level = None,
        stop_at_unet_number = None,
        return_all_unet_outputs = False,
        return_pil_images = False,
        device = None,
        use_tqdm = True
    ):
        device = default(device, self.device)
        self.reset_unets_all_one_device(device = device)

        # cond_images = maybe(cast_uint8_images_to_float)(cond_images)

        if conditional_input is not None:
            # input_rgb, input_cameras = conditional_input
            # conditional_embeds = self.encode_conditional(input_rgb, cameras=input_cameras)
            conditional_embeds = self.encode_conditional(conditional_input)

        if not self.unconditional:
            conditional_masks = default(conditional_masks, lambda: torch.any(conditional_embeds != 0., dim = -1))

        if not self.unconditional:
            batch_size = conditional_embeds.shape[0]

        assert not (self.conditional and not exists(conditional_embeds)), 'text or text encodings must be passed into imagen if specified'
        assert not (not self.conditional and exists(conditional_embeds)), 'imagen specified not to be conditioned on text, yet it is presented'
        assert not (exists(conditional_embeds) and conditional_embeds.shape[-1] != self.conditional_embed_dim), f'invalid text embedding dimension being passed in (should be {self.conditional_embed_dim})'

        assert not (exists(inpaint_images) ^ exists(inpaint_masks)),  'inpaint images and masks must be both passed in to do inpainting'

        outputs = []

        is_cuda = next(self.parameters()).is_cuda
        device = next(self.parameters()).device

        lowres_sample_noise_level = default(lowres_sample_noise_level, self.lowres_sample_noise_level)

        num_unets = len(self.unets)
        cond_scale = cast_tuple(cond_scale, num_unets)

        assert not (self.is_video and not exists(video_frames)), 'video_frames must be passed in on sample time if training on video'

        frame_dims = (video_frames,) if self.is_video else tuple()

        for unet_number, unet, channel, image_size, noise_scheduler, pred_objective, dynamic_threshold, unet_cond_scale in tqdm(zip(range(1, num_unets + 1), self.unets, self.sample_channels, self.image_sizes, self.noise_schedulers, self.pred_objectives, self.dynamic_thresholding, cond_scale), disable = not use_tqdm):

            # context = self.one_unet_in_gpu(unet = unet) if is_cuda else nullcontext()
            context = nullcontext()

            with context:
                lowres_cond_img = lowres_noise_times = None
                shape = (batch_size, channel, *frame_dims, image_size, image_size)

                if unet.lowres_cond:
                    lowres_noise_times = self.lowres_noise_schedule.get_times(batch_size, lowres_sample_noise_level, device = device)

                    lowres_cond_img = self.resize_to(img, image_size)

                    lowres_cond_img = self.normalize_img(lowres_cond_img)
                    lowres_cond_img, _ = self.lowres_noise_schedule.q_sample(x_start = lowres_cond_img, t = lowres_noise_times, noise = torch.randn_like(lowres_cond_img))

                shape = (batch_size, self.channels, *frame_dims, image_size, image_size)

                img = self.p_sample_loop(
                    unet,
                    shape,
                    conditional_embeds = conditional_embeds,
                    conditional_mask = conditional_masks,
                    cond_images = cond_images,
                    inpaint_images = inpaint_images,
                    inpaint_masks = inpaint_masks,
                    inpaint_resample_times = inpaint_resample_times,
                    cond_scale = unet_cond_scale,
                    lowres_cond_img = lowres_cond_img,
                    lowres_noise_times = lowres_noise_times,
                    noise_scheduler = noise_scheduler,
                    pred_objective = pred_objective,
                    dynamic_threshold = dynamic_threshold,
                    use_tqdm = use_tqdm
                )

                outputs.append(img)

            if exists(stop_at_unet_number) and stop_at_unet_number == unet_number:
                break

        output_index = -1 if not return_all_unet_outputs else slice(None) # either return last unet output or all unet outputs

        if not return_pil_images:
            return outputs[output_index]

        if not return_all_unet_outputs:
            outputs = outputs[-1:]

        assert not self.is_video, 'converting sampled video tensor to video file is not supported yet'

        pil_images = list(map(lambda img: list(map(T.ToPILImage(), img.unbind(dim = 0))), outputs))

        return pil_images[output_index] # now you have a bunch of pillow images you can just .save(/where/ever/you/want.png)

    def p_losses(
        self,
        unet,
        x_start,
        times,
        *,
        noise_scheduler,
        lowres_cond_img = None,
        lowres_aug_times = None,
        conditional_embeds = None,
        conditional_mask = None,
        cond_images = None,
        noise = None,
        times_next = None,
        pred_objective = 'noise',
        p2_loss_weight_gamma = 0.,
        random_crop_size = None,
        loss_mask = None
    ):
        is_video = x_start.ndim == 5

        noise = default(noise, lambda: torch.randn_like(x_start))

        # normalize to [-1, 1]

        x_start = self.normalize_img(x_start)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        x_noisy, log_snr = noise_scheduler.q_sample(x_start = x_start, t = times, noise = noise)

        # also noise the lowres conditioning image
        # at sample time, they then fix the noise level of 0.1 - 0.3

        lowres_cond_img_noisy = None
        if exists(lowres_cond_img):
            lowres_aug_times = default(lowres_aug_times, times)
            lowres_cond_img_noisy, _ = self.lowres_noise_schedule.q_sample(x_start = lowres_cond_img, t = lowres_aug_times, noise = torch.randn_like(lowres_cond_img))

        # get prediction

        pred = unet.forward(
            x_noisy,
            noise_scheduler.get_condition(times),
            conditional_embeds = conditional_embeds,
            conditional_mask = conditional_mask,
            cond_images = cond_images,
            lowres_noise_times = self.lowres_noise_schedule.get_condition(lowres_aug_times),
            lowres_cond_img = lowres_cond_img_noisy,
            cond_drop_prob = self.cond_drop_prob,
        )

        # prediction objective

        if pred_objective == 'noise':
            target = noise
        elif pred_objective == 'x_start':
            target = x_start
        else:
            raise ValueError(f'unknown objective {pred_objective}')

        # losses
        if loss_mask is not None:
            pred = pred * loss_mask
            target = target * loss_mask
        losses = self.loss_fn(pred, target, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        # p2 loss reweighting

        if p2_loss_weight_gamma > 0:
            loss_weight = (self.p2_loss_weight_k + log_snr.exp()) ** -p2_loss_weight_gamma
            losses = losses * loss_weight

        return losses.mean()

    @torch.no_grad()
    def forward_noloss(
        self,
        images,
        unet: Unet = None,
        conditional_input = None,
        conditional_embeds = None,
        conditional_masks = None,
        unet_number = None,
        cond_images = None,
        loss_mask = None,
        return_noise = False,
        min_thres = 0.0,
        max_thres = 0.999,
        method = 'onestep',
    ):

        #@ DEFINE DEFAULT UNET
        unet_number = default(unet_number, 1)
        unet_index = unet_number - 1
        unet = default(unet, lambda: self.get_unet(unet_number))

        #@ GET SCHEDULERS
        noise_scheduler      = self.noise_schedulers[unet_index]
        p2_loss_weight_gamma = self.p2_loss_weight_gamma[unet_index]
        pred_objective       = self.pred_objectives[unet_index]
        target_image_size    = self.image_sizes[unet_index]
        random_crop_size     = self.random_crop_sizes[unet_index]
        prev_image_size      = self.image_sizes[unet_index - 1] if unet_index > 0 else None

        b, c, *_, h, w, device, is_video = *images.shape, images.device, images.ndim == 5


        #@ PROCESS CONDITIONAL INPUT
        if conditional_input is not None:
            # input_rgb, input_cameras = conditional_input
            # conditional_embeds = self.encode_conditional(input_rgb, cameras=input_cameras)
            conditional_embeds = self.encode_conditional(conditional_input)

        if not self.unconditional:
            conditional_masks = default(conditional_masks, lambda: torch.any(conditional_embeds != 0., dim = -1))

        images = self.resize_to(images, target_image_size)

        if method == 'onestep':
            #@ SAMPLE TIMES
            times = noise_scheduler.sample_random_times_bounded(b, min_thres=min_thres, max_thres=max_thres, device = device)

            #@ PREPARE FOR FORWARD PASS
            x_start = self.normalize_img(images)
            noise = None
            noise = default(noise, lambda: torch.randn_like(x_start))
            x_noisy, log_snr = noise_scheduler.q_sample(x_start = x_start, t = times, noise = noise)

            #@ Unet
            pred = unet.forward(
                x_noisy,
                noise_scheduler.get_condition(times),
                conditional_embeds = conditional_embeds,
                conditional_mask = conditional_masks,
                cond_images = cond_images,
                lowres_noise_times = None,
                lowres_cond_img = None,
                cond_drop_prob = self.cond_drop_prob,
            )

            pred_x0 = noise_scheduler.predict_start_from_noise(x_noisy, t = times, noise = pred)

            if self.clip_output:
                pred_x0.clamp_(-self.clip_value, self.clip_value)

        else:
            raise NotImplementedError

        if return_noise:
            alpha_cumprod = torch.sigmoid(log_snr)
            return pred, pred_x0, x_noisy, noise, alpha_cumprod
        return pred

    def forward(
        self,
        images,
        unet: Unet = None,
        conditional_input = None,
        conditional_embeds = None,
        conditional_masks = None,
        unet_number = None,
        cond_images = None,
        loss_mask = None
    ):
        assert images.shape[-1] == images.shape[-2], f'the images you pass in must be a square, but received dimensions of {images.shape[2]}, {images.shape[-1]}'
        assert not (len(self.unets) > 1 and not exists(unet_number)), f'you must specify which unet you want trained, from a range of 1 to {len(self.unets)}, if you are training cascading DDPM (multiple unets)'
        unet_number = default(unet_number, 1)
        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, 'you can only train on unet #{self.only_train_unet_number}'

        # images = cast_uint8_images_to_float(images)
        # cond_images = maybe(cast_uint8_images_to_float)(cond_images)

        unet_index = unet_number - 1
        
        unet = default(unet, lambda: self.get_unet(unet_number))

        noise_scheduler      = self.noise_schedulers[unet_index]
        p2_loss_weight_gamma = self.p2_loss_weight_gamma[unet_index]
        pred_objective       = self.pred_objectives[unet_index]
        target_image_size    = self.image_sizes[unet_index]
        random_crop_size     = self.random_crop_sizes[unet_index]
        prev_image_size      = self.image_sizes[unet_index - 1] if unet_index > 0 else None

        b, c, *_, h, w, device, is_video = *images.shape, images.device, images.ndim == 5

        check_shape(images, 'b c ...', c = self.channels)
        assert h >= target_image_size and w >= target_image_size

        frames = images.shape[2] if is_video else None

        times = noise_scheduler.sample_random_times(b, device = device)
        
        if conditional_input is not None:
            # input_rgb, input_cameras = conditional_input
            # conditional_embeds = self.encode_conditional(input_rgb, cameras=input_cameras)
            conditional_embeds = self.encode_conditional(conditional_input)

        if not self.unconditional:
            conditional_masks = default(conditional_masks, lambda: torch.any(conditional_embeds != 0., dim = -1))

        assert not (self.conditional and not exists(conditional_embeds)), 'text or text encodings must be passed into decoder if specified'
        assert not (not self.conditional and exists(conditional_embeds)), 'decoder specified not to be conditioned on text, yet it is presented'

        assert not (exists(conditional_embeds) and conditional_embeds.shape[-1] != self.conditional_embed_dim), f'invalid text embedding dimension being passed in (should be {self.conditional_embed_dim})'

        lowres_cond_img = lowres_aug_times = None
        if exists(prev_image_size):
            lowres_cond_img = self.resize_to(images, prev_image_size, clamp_range = self.input_image_range)
            lowres_cond_img = self.resize_to(lowres_cond_img, target_image_size, clamp_range = self.input_image_range)

            if self.per_sample_random_aug_noise_level:
                lowres_aug_times = self.lowres_noise_schedule.sample_random_times(b, device = device)
            else:
                lowres_aug_time = self.lowres_noise_schedule.sample_random_times(1, device = device)
                lowres_aug_times = repeat(lowres_aug_time, '1 -> b', b = b)

        images = self.resize_to(images, target_image_size)

        return self.p_losses(unet, images, times, conditional_embeds = conditional_embeds, conditional_mask = conditional_masks, cond_images = cond_images, noise_scheduler = noise_scheduler, lowres_cond_img = lowres_cond_img, lowres_aug_times = lowres_aug_times, pred_objective = pred_objective, p2_loss_weight_gamma = p2_loss_weight_gamma, random_crop_size = random_crop_size, loss_mask=loss_mask)
