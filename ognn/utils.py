# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from typing import Optional
from ocnn.utils import broadcast, scatter_add
from ocnn.octree import key2xyz
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches


def key2coords(keys: torch.Tensor, depth: int):
    x, y, z, _ = key2xyz(keys, depth)
    return torch.stack([x, y, z], dim=-1)


def plot_graph_matplot(row: torch.Tensor, col: torch.Tensor, edge_dir: torch.Tensor, filename: str):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 绘制所有点
    all_points = torch.cat([row, col], dim=0)
    ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c="blue")

    # 绘制边
    color_map = cm.get_cmap("tab10", 7)
    for r, c, direction in zip(row, col, edge_dir):
        vec = c - r
        ax.quiver(
            r[0].item(),
            r[1].item(),
            r[2].item(),  # 起点
            vec[0].item(),
            vec[1].item(),
            vec[2].item(),  # 向量方向
            length=1.0,  # 保持原向量长度
            normalize=False,  # 标准化长度，使用 length 缩放
            arrow_length_ratio=0.1,  # 箭头占向量长度比例
            color=color_map(direction.item() // 2),
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    patches = [mpatches.Patch(color=color_map(i), label=i) for i in range(3)]
    ax.legend(handles=patches, loc="upper right")
    plt.savefig(filename)


def visualize_graph(octree, depth: int, filename: str):
    graph = octree.graphs[depth]
    keys = graph.key & 0xFFFFFFFFFFFF
    keys_aligned = keys << (3 * (depth - graph.node_depth))
    coords = key2coords(keys_aligned, depth)
    rows, cols = graph.edge_idx
    row_coords = coords[rows]
    row_keys = keys_aligned[rows].view(-1, 1)
    col_keys = keys_aligned[cols].view(-1, 1)
    col_coords = coords[cols]
    # print(torch.cat((row_coords, col_coords, row_keys, col_keys), dim=-1))
    plot_graph_matplot(
        row_coords + (2.0 ** (depth - graph.node_depth - 1).view(-1, 1))[rows],
        col_coords + (2.0 ** (depth - graph.node_depth - 1).view(-1, 1))[cols],
        graph.edge_dir,
        filename,
    )


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
