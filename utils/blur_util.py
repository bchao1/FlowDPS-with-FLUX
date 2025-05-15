import torch
from torch import nn
import numpy as np
import scipy
import imageio
from utils.motionblur import Kernel as MotionKernel

class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size//2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            sigma = 3.0
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            ker1d = []
            for k in range(-self.kernel_size//2, self.kernel_size//2):
                ker1d.append(pdf(k))
            ker1d = torch.Tensor(ker1d).to(self.device)
            ker2d = torch.ger(ker1d, ker1d)
            ker2d = ker2d / ker2d.sum()
            ker2d = np.ones((self.kernel_size, self.kernel_size))
            ker2d = ker2d / ker2d.sum()
            ker2d = torch.from_numpy(ker2d).to(self.device)
            self.k = ker2d
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = MotionKernel(size=(self.kernel_size, self.kernel_size), intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k