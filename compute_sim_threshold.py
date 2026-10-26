import argparse

import torch
import torch.nn.functional as F
import numpy as np

def _energy_map(trigger):
    """To (H, W) energy map."""
    if trigger.dim() == 3:
        return trigger.abs().mean(dim=0)
    return trigger.abs()


def _clip_topk(trigger, k):
    """
    Zero out all but the top-k pixels (by energy) in a trigger tensor.
    """
    energy = _energy_map(trigger) # (H, W)
    flat = energy.flatten()
    if flat.numel() <= k:
        return trigger.clone()
    threshold = torch.topk(flat, k).values[-1]
    mask = (energy >= threshold).float() # (H, W)
    if trigger.dim() == 3:
        return trigger * mask.unsqueeze(0)
    return trigger * mask


def ssim_score(trigger1, trigger2, k=49, window_size=11):

    if trigger1 is None or trigger2 is None:
        return False

    t1 = _clip_topk(trigger1, k)
    t2 = _clip_topk(trigger2, k)

    e1 = _energy_map(t1).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    e2 = _energy_map(t2).unsqueeze(0).unsqueeze(0)

    # Dynamic-range-calibrated SSIM constants (Wang et al., 2004)
    L = max(e1.max().item(), e2.max().item())
    L = max(L, 1e-8)  # avoid division by zero for all-zero maps
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size, device=trigger1.device) / (window_size ** 2)

    mu1 = F.conv2d(e1, kernel, padding=pad)
    mu2 = F.conv2d(e2, kernel, padding=pad)
    sigma1_sq = F.conv2d(e1 ** 2, kernel, padding=pad) - mu1 ** 2
    sigma2_sq = F.conv2d(e2 ** 2, kernel, padding=pad) - mu2 ** 2
    sigma12   = F.conv2d(e1 * e2, kernel, padding=pad) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    score = ssim_map.mean().item()
    return score
    
    
def random_trigger(H, W, sigma):
    noise = torch.randn(1, 1, H, W)
    kernel = max(3, int(6 * sigma) | 1)
    padding = kernel // 2 # to ensure output is same size
    
    x = torch.arange(kernel).float() - padding
    gaussian1d = torch.exp(-x**2 / (2*sigma**2))
    gaussian2d = gaussian1d.unsqueeze(1) * gaussian1d.unsqueeze(0)
    full_gaussian = (gaussian2d / gaussian2d.sum()).unsqueeze(0).unsqueeze(0) # normalize and move to 4d
    
    smoothed_noise = F.conv2d(noise, full_gaussian, padding=padding)
    return smoothed_noise.squeeze()
    
if __name__ == "__main__":
    # (H, W, ssim_window_size, ssim_topk) for each dataset
    DATASET_PARAMS = {
        "cifar10": (32, 32, 11, 51),
        "femnist": (28, 28, 9, 39),
        "tinyimagenet": (64, 64, 19, 205),
    }
    parser = argparse.ArgumentParser(description="SSIM statistics between random triggers")
    parser.add_argument("--dataset", choices=DATASET_PARAMS, default="cifar10")
    args = parser.parse_args()

    H, W, w, k = DATASET_PARAMS[args.dataset]
    sigma = 2
    M = 10000

    scores = []
    for _ in range(M):
        t1 = random_trigger(H, W, sigma)
        t2 = random_trigger(H, W, sigma)
        scores.append(ssim_score(t1, t2, k=k, window_size=w))
    scores = np.array(scores)
    print(f"Mean: {np.mean(scores)}, std: {np.std(scores)}")
    print(f"q95: {np.percentile(scores, 95)}, q99: {np.percentile(scores, 99)}")


