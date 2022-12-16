'''
Fast PNDM PLMS Sampler
#@ FROM https://github.com/CompVis/stable-diffusion
'''
import torch
import numpy as np
from tqdm import tqdm
from external.imagen_pytorch import GaussianDiffusionContinuousTimes, right_pad_dims_to
from sparsefusion.vldm import DDPM
from einops import rearrange


class PLMSSampler():

    def __init__(self, diffusion: DDPM, plms_steps=100):

        self.diffusion = diffusion
        self.plms_steps = plms_steps

    @torch.no_grad()
    def sample(self,
        image=None,
        max_thres=.999,
        cond_images=None,
        cond_scale=1.0,
        use_tqdm=True,
        return_noise=False,
        **kwargs
    ):
        '''
        Single UNet PLMS Sampler
        '''
        outputs = []
        batch_size = cond_images.shape[0]
        shape = (batch_size, self.diffusion.sample_channels[0], self.diffusion.image_sizes[0], self.diffusion.image_sizes[0])
        img, x_noisy, noise, alpha_cumprod = self.plms_sample_loop(
                    self.diffusion.unets[0],
                    image = image,
                    shape = shape,
                    cond_images = cond_images,
                    cond_scale = cond_scale,
                    noise_scheduler = self.diffusion.noise_schedulers[0],
                    pred_objective = self.diffusion.pred_objectives[0],
                    dynamic_threshold = self.diffusion.dynamic_thresholding[0],
                    use_tqdm = use_tqdm,
                    max_thres = max_thres,
                )
        outputs.append(img)
        if not return_noise:
            return outputs[-1]
        return outputs[-1], x_noisy, noise, alpha_cumprod

    @torch.no_grad()
    def plms_sample_loop(self,
        unet,
        image,
        shape,
        cond_images,
        cond_scale,
        noise_scheduler,
        pred_objective,
        dynamic_threshold,
        use_tqdm,
        max_thres = None,
    ):
        '''
        Sampling loop
        '''
        batch = shape[0]
        device = self.diffusion.device

        if image is None:
            image = torch.randn(shape, device = device)
        else:
            assert(max_thres is not None)


        old_eps = []
        
        if max_thres >= .99:
            noise_scheduler_short = GaussianDiffusionContinuousTimes(noise_schedule='cosine', timesteps=self.plms_steps)
            timesteps = noise_scheduler_short.get_sampling_timesteps(batch, device=device)
            noise = torch.randn_like(image)
            x_noisy, log_snr= noise_scheduler_short.q_sample(image, t=max_thres, noise=noise)
            img = image
        else:
            n_steps = min(int(max_thres * self.plms_steps * 2), self.plms_steps)
            # n_steps = 50
            noise_scheduler_short = GaussianDiffusionContinuousTimes(noise_schedule='cosine', timesteps=self.plms_steps)
            timesteps = noise_scheduler_short.get_sampling_timesteps_custom(batch, device=device, max_thres=max_thres, n_steps=n_steps)
            noise = torch.randn_like(image)
            img, log_snr = noise_scheduler_short.q_sample(image, t=max_thres, noise=noise)
            x_noisy = img

        for times, times_next in tqdm(timesteps, desc = 'sampling loop time step', total = len(timesteps), disable = not use_tqdm):
            is_last_timestep = times_next == 0
            outs = self.p_sample(
                unet,
                img,
                times,
                t_next = times_next,
                cond_images = cond_images,
                cond_scale = cond_scale,
                noise_scheduler = noise_scheduler,
                pred_objective = pred_objective,
                dynamic_threshold = dynamic_threshold,
                old_eps = old_eps
            )
            img, pred_x0, e_t = outs
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)

        if self.diffusion.clip_output:
            img.clamp_(-self.diffusion.clip_value, self.diffusion.clip_value)

        unnormalize_img = self.diffusion.unnormalize_img(img)
        alpha_cumprod = torch.sigmoid(log_snr)
        return unnormalize_img, x_noisy, noise, alpha_cumprod

    @torch.no_grad()
    def p_sample(self,
        unet,
        x,
        t,
        t_next,
        cond_images,
        cond_scale,
        noise_scheduler,
        pred_objective,
        dynamic_threshold,
        old_eps
    ):

        b, *_, device = *x.shape, x.device
        _, _, e_t = self.get_model_output(unet, x, t, t_next, cond_images, cond_scale, noise_scheduler, pred_objective, dynamic_threshold)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            # x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            x_prev, pred_x0, _ = self.get_model_output(unet, x, t, t_next, cond_images, cond_scale, noise_scheduler, pred_objective, dynamic_threshold, pred_e = e_t)
            # e_t_next = get_model_output(x_prev, t_next)
            _, _, e_t_next = self.get_model_output(unet, x_prev, t_next, t_next, cond_images, cond_scale, noise_scheduler, pred_objective, dynamic_threshold)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0, _ = self.get_model_output(unet, x, t, t_next, cond_images, cond_scale, noise_scheduler, pred_objective, dynamic_threshold, pred_e = e_t_prime)

        return x_prev, pred_x0, e_t

    def get_model_output(self,
        unet,
        x,
        t,
        t_next,
        cond_images,
        cond_scale,
        noise_scheduler,
        pred_objective,
        dynamic_threshold,
        pred_e = None,
    ):
        assert(pred_objective == 'noise')
        b, *_, device = *x.shape, x.device


        #@ PRED EPS
        if pred_e is None:
            pred = unet.forward_with_cond_scale(x, noise_scheduler.get_condition(t), cond_images = cond_images, cond_scale = cond_scale)
            pred_e = pred
        else:
            pred = pred_e


        #@ PREDICT X_0
        if pred_objective == 'noise':            
            x_start = noise_scheduler.predict_start_from_noise(x, t = t, noise = pred)
        elif pred_objective == 'x_start':
            x_start = pred
        else:
            raise ValueError(f'unknown objective {pred_objective}')

        #@ CLIP X_0
        if self.diffusion.clip_output:
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
                x_start.clamp_(-self.diffusion.clip_value, self.diffusion.clip_value)

        #@ USE Q_POSTERIOR TO GET X_PREV
        model_mean, _, model_log_variance = noise_scheduler.q_posterior(x_start = x_start, x_t = x, t = t, t_next = t_next)
        noise = torch.randn_like(x)
        is_last_sampling_timestep = (t_next == 0) if isinstance(noise_scheduler, GaussianDiffusionContinuousTimes) else (t == 0)
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        x_prev = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        return x_prev, x_start, pred_e
