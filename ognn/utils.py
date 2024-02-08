# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from typing import Optional
from ocnn.utils import broadcast, scatter_add


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
  out = scatter_add(src, index, dim, out, dim_size)
  dim_size = out.size(dim)

  index_dim = dim
  if index_dim < 0:
    index_dim = index_dim + src.dim()
  if index.dim() <= index_dim:
    index_dim = index.dim() - 1

  ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
  count = scatter_add(ones, index, index_dim, None, dim_size)
  count[count < 1] = 1
  count = broadcast(count, out, dim)
  if out.is_floating_point():
    out.true_divide_(count)
  else:
    out.div_(count, rounding_mode='floor')
  return out


def spmm(index, value, m, n, matrix):
  r'''Matrix product of sparse matrix with dense matrix.

  Args:
      index (:class:`LongTensor`): The index tensor of sparse matrix.
      value (:class:`Tensor`): The value tensor of sparse matrix.
      m (int): The first dimension of corresponding dense matrix.
      n (int): The second dimension of corresponding dense matrix.
      matrix (:class:`Tensor`): The dense matrix.

  :rtype: :class:`Tensor`
  '''

  assert n == matrix.size(-2)

  row, col = index
  matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

  out = matrix.index_select(-2, col)
  out = out * value.unsqueeze(-1)
  out = scatter_add(out, row, dim=-2, dim_size=m)

  return out
