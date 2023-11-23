import numpy as np
from utils import to_y_channel


def calculate_mse(img1, img2, test_y_channel=False):

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    return np.mean((img1 - img2) ** 2)