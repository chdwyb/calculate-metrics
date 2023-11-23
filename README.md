This repository can be employed to calculate five metrics for evaluating the quality of images, namely peak signal-to-noise ratio (PNSR), structural similarity (SSIM), mean squared error (MSE), natural image quality evaluator (NIQE) and learned perpetual image patch similarity (LPIPS). If you find this repository helpful in your research, please consider giving this repository a star :star2:.

#### Dependencies

```python
Pillow 9.2.0
opencv-python 4.8.0.76
lpips 0.1.4
numpy 1.23.1
scipy 1.9.1
torch 1.9.0
torchvision 0.10.0
```

#### Usage

Modify the following two paths in the `main.py` file, and then run it.

```python
path_result = 'E://0000ceshi/result'
path_target = 'E://0000ceshi/target'
```

