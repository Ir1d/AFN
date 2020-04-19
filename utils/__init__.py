#!/usr/bin/env python3
__all__ = ["load_module", "SOBEL", "Sobel"]

import loguru
import copy
import sys
import os
import pkgutil
import keyword
import importlib
import importlib.machinery
import torch
import torch.nn as nn
import numpy as np
from kornia.filters.sobel import spatial_gradient
from albumentations.core.transforms_interface import ImageOnlyTransform


def load_module(fpath):
    fpath = os.path.realpath(fpath)
    mod_name = []
    for i in fpath.split(os.path.sep):
        v = str()
        for j in i:
            if not j.isidentifier() and not j.isdigit():
                j = '_'
            v += j
        if not v.isidentifier() or keyword.iskeyword(v):
            v = '_' + v
        mod_name.append(v)
    mod_name = '_'.join(mod_name)
    if mod_name in sys.modules:  # return if already loaded
        return sys.modules[mod_name]
    mod_dir = os.path.dirname(fpath)
    sys.path.append(mod_dir)
    old_mod_names = set(sys.modules.keys())
    try:
        final_mod = importlib.machinery.SourceFileLoader(
            mod_name, fpath).load_module()
    finally:
        sys.path.remove(mod_dir)
    sys.modules[mod_name] = final_mod
    return final_mod


class Sobel(nn.Module):
    r"""Computes the Sobel operator and returns the magnitude per channel.

    Return:
        torch.Tensor: the sobel edge gradient maginitudes map.

    Args:
        normalized (bool): if True, L1 norm of the kernel is set to 1.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = kornia.filters.Sobel()(input)  # 1x3x4x4
    """

    def __init__(self, axis=None,
                 normalized: bool = True) -> None:
        super(Sobel, self).__init__()
        assert axis in {None, 2, 3, -1, -2}
        self.normalized: bool = normalized
        self.axis = axis

    def __repr__(self) -> str:
        return self.__class__.__name__ + '('\
            'normalized=' + str(self.normalized) + ')'

    def forward(self, input: torch.Tensor, axis=None) -> torch.Tensor:  # type: ignore
        if axis is None:
            axis = self.axis
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # comput the x/y gradients
        edges: torch.Tensor = spatial_gradient(input,
                                               normalized=self.normalized)

        # unpack the edges
        gx: torch.Tensor = edges[:, :, 0]
        gy: torch.Tensor = edges[:, :, 1]
        # compute gradient maginitude
        magnitude: torch.Tensor = torch.sqrt(gx * gx + gy * gy)
        if axis is None:
            return magnitude
        elif axis in {3, -1}:
            return gx
        elif axis in {2, -2}:
            return gy
# vim: ts=4 sw=4 sts=4 expandtab


def SOBEL(x, axis=None):
    return Sobel(axis=axis)(x)


class ProjectableTensorDict(dict):
    def apply(self, func_name, *args, **kwargs):
        rtn = copy.copy(self)
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                rtn[k] = getattr(v, func_name)(*args, **kwargs)
            else:
                assert type(v) in {str} or k == "name", "%s %s %s" % (
                    k, type(v), v)
        return rtn


class RandomSwapChannel(ImageOnlyTransform):
    def apply(self, x, **params):
        return np.ascontiguousarray(x[:, :, ::-1])
