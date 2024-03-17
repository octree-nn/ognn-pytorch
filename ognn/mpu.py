# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------


import ocnn
import torch
import torch.nn

from ocnn.octree import Octree, xyz2key
from ocnn.utils import scatter_add, range_grid
from easydict import EasyDict as edict


class NeuralMPU:

  def __init__(self):
    self.kNN = 8

  def linear_basis(self, x: torch.Tensor):
    return 1.0 - x.abs()

  def perpare(self, octree: Octree, depth: int, pts: torch.Tensor):
    r""" Prepare the data for the Multi-Point Upsampling (MPU) operation. """

    # compute the coordinates of neighboring octree nodes
    scale = 2 ** depth
    mask = range_grid(0, 1, pts.device)
    xyzf, ids = torch.split(pts, [3, 1], 1)
    xyzf = (xyzf + 1.0) * (scale / 2.0)    # [-1, 1] -> [0, scale]
    xyzf = xyzf - 0.5                      # the code is defined on the center
    xyzi = torch.floor(xyzf).detach()      # the integer part (N, 3), use floor
    corners = xyzi.unsqueeze(1) + mask  # (N, 8, 3)
    coordsf = xyzf.unsqueeze(1) - corners    # (N, 8, 3), in [-1.0, 1.0]

    # coorers -> key -> the indices of neighboring octree nodes
    xyz = corners.view(-1, 3)
    ids = ids.detach().repeat(1, self.kNN).view(-1)       # (N, 8, 1) -> (N*8, )
    key = xyz2key(xyz[:, 0], xyz[:, 1], xyz[:, 2], ids, depth)
    idx = octree.search_key(key, depth)

    # corners -> in-bound flags
    inbound = torch.logical_and(corners > -1, corners < scale)
    inbound = torch.all(inbound, dim=-1).view(-1)
    valid = torch.logical_and(idx > -1, inbound)  # in-bound and idx > -1

    # the indices of the pts itself
    ids = torch.arange(pts.size(0), device=pts.device, dtype=torch.long)
    ids = ids.unsqueeze(-1).repeat(1, self.kNN).view(-1)
    ids = ids[valid]

    # remove invalid pts
    idx = idx[valid].long()               # (N*8, )   -> (N', )
    coordsf = coordsf.view(-1, 3)[valid]  # (N, 8, 3) -> (N', 3)

    # bspline weights
    weights = self.linear_basis(coordsf)                # (N', 3)
    weights = torch.prod(weights, axis=-1).view(-1)     # (N', )

    # Here, the scale factor `2**(depth - 6)` is used to emphasize high-resolution
    # basis functions. Tune this factor further if needed! !!! NOTE !!!
    # weights = weights * 2**(depth - 6)                 # used for shapenet
    weights = weights * (depth**2 / 50)                  # testing

    # rescale back the original scale.
    # after recaling, the coordsf is in the same scale as `pts``
    coordsf = coordsf * (2.0 / scale)   # [-1.0, 1.0] -> [-2.0/scale, 2.0/scale]
    return edict({'ids': ids, 'idx': idx, 'xyzf': coordsf, 'weights': weights})

  def compute(self, pts: torch.Tensor, feature: torch.Tensor, octree: Octree,
              mpus: edict, depth_start: int, depth_end: int):
    npt = pts.size(0)
    indices, weights, xyzfs = [], [], []
    nnum_cum = ocnn.utils.cumsum(octree.nnum, dim=0, exclusive=True)
    for d in range(depth_start, depth_end+1):
      idxd = mpus[d].idx
      idsd = mpus[d].ids
      xyzfd = mpus[d].xyzf
      weightd = mpus[d].weights

      if d < depth_end:
        child = octree.children[d]
        leaf = child[idxd] < 0  # keep only leaf nodes
        idsd = idsd[leaf]
        idxd = idxd[leaf]
        weightd = weightd[leaf]
        xyzfd = xyzfd[leaf]

      idxd = idxd + (nnum_cum[d] - nnum_cum[depth_start])
      indices.append(torch.stack([idsd, idxd], dim=1))
      weights.append(weightd)
      xyzfs.append(xyzfd)

    indices = torch.cat(indices, dim=0).t()
    weights = torch.cat(weights, dim=0)
    xyzfs = torch.cat(xyzfs, dim=0)
    output = self.spmm(indices, weights, npt, feature.size(0), feature, xyzfs)
    return output

  def spmm(self, index, value, m, n, matrix, xyzf):
    row, col = index
    assert n == matrix.size(-2)
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix.index_select(-2, col)
    ones = torch.ones((xyzf.shape[0], 1), device=xyzf.device)
    xyz1 = torch.cat([xyzf, ones], dim=1)
    out = torch.sum(out * xyz1, dim=1, keepdim=True)
    out = out * value.unsqueeze(-1)
    out = scatter_add(out, row, dim=-2, dim_size=m)

    ones = torch.ones(n, 1, device=matrix.device)
    norm = ones.index_select(0, col)
    norm = norm * value.unsqueeze(-1)
    norm = scatter_add(norm, row, dim=-2, dim_size=m)

    out = torch.div(out, norm + 1e-8).squeeze()
    return out

  def setup(self, features: torch.Tensor, octree: Octree, depth_end: int):
    self.buffer = (features, octree, depth_end)

  def __call__(self, pts: torch.Tensor):
    features, octree, depth_end = self.buffer

    fvals, mpus = dict(), dict()
    depth_start = octree.full_depth
    for d in range(depth_start, depth_end+1):
      mpus[d] = self.perpare(octree, d, pts)
      fvals[d] = self.compute(pts, features[d], octree, mpus, depth_start, d)
    return fvals
