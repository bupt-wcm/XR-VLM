import torch

import cv2
import numpy as np

import matplotlib.pyplot as plt


def norm_img(t, ):
    def norm_ip(img, low, high):
        img = img.clone().clamp(min=low, max=high)
        img = img.sub(low).div(max(high - low, 1e-5))
        return img

    return norm_ip(t, float(t.min()), float(t.max()))


def show_img(img:torch.Tensor, is_norm, new_figure=True, figsize=(8, 8), show=True):
    if is_norm:
        img = norm_img(img)
    if new_figure:
        fig = plt.figure(figsize=figsize)
    if img.dim() == 3:
        plt.imshow(img.permute(1, 2, 0))
    else:
        plt.imshow(img)
    plt.axis('off')
    if show:
        plt.tight_layout()
        plt.show()


def show_imgs(imgs: torch.Tensor, is_norm, nrows, ncols, figsize=(8, 8)):
    assert imgs.dim() == 4 and imgs.shape[1] in [1, 3], \
        'imgs should be a 4-dim tensor, and its 1-st shape should 1 or 3'
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    if is_norm:
        imgs = [norm_img(img) for img in imgs]

    for idx in range(nrows * ncols):
        if idx >= len(imgs):
            break
        axes[idx].imshow(imgs[idx].permute(1, 2, 0))
        axes[idx].axis('off')
    plt.tight_layout()
    plt.show()


def show_maps(maps: torch.Tensor, nrows, ncols, figsize=(8, 8)):
    """
    utils to show map with single channel, maps can be single or multiple maps
    """
    assert maps.dim() in [2, 3], \
        'tensor dim must be in [2 (single map), 3 (batch of single map), 4 (batch of images)]'
    if maps.dim() == 2:
        maps = maps.unsqueeze(0)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    for idx in range(nrows * ncols):
        axes[idx].imshow(maps[idx])
        axes[idx].axis('off')
    plt.tight_layout()
    plt.show()


def map_to_img(m, img, is_norm=True, scale=(0.3, 0.5)):
    if is_norm:
        img = norm_img(img)
    m = m - torch.min(m)
    m = m / torch.max(m)
    img = cv2.cvtColor(np.uint8(img.permute(1, 2, 0).numpy() * 255.), cv2.COLOR_RGB2BGR)
    height, width, _ = img.shape
    m = torch.stack([m, m, m], dim=-1)
    heatmap = cv2.applyColorMap(
        cv2.resize(np.uint8(m.numpy() * 255.), (width, height)), cv2.COLORMAP_JET
    )
    result = heatmap * scale[0] + img * scale[1]
    result = cv2.cvtColor(np.uint8(result), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(result).permute(2, 0, 1)