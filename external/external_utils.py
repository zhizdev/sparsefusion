"""
A wrapper class for the perceptual deep feature loss.

Reference:
    Richard Zhang et al. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. (CVPR 2018).
"""
import lpips
import torch.nn as nn


class PerceptualLoss(nn.Module):
    def __init__(self, net, device):
        super().__init__()
        self.model = lpips.LPIPS(net=net, verbose=False).to(device)
        self.device = device

    def get_device(self, default_device=None):
        """
        Returns which device module is on, assuming all parameters are on the same GPU.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            return default_device

    def __call__(self, pred, target, normalize=True):
        """
        Pred and target are Variables.
        If normalize is on, scales images between [-1, 1]
        Assumes the inputs are in range [0, 1].
        B 3 H W
        """
        if pred.shape[1] != 3:
            pred = pred.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)
        # print(pred.shape, target.shape)
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1

        # temp_device = pred.device
        # device = self.get_device(temp_device)

        device = self.device

        pred = pred.to(device).float()
        target = target.to(device)
        dist = self.model.forward(pred, target)
        return dist.to(device)
