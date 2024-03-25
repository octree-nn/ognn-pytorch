# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import ocnn

from typing import Optional
from ocnn.octree import Octree, key2xyz, xyz2key


class Graph:

  def __init__(self, **kwargs):
    super().__init__()
    self.key_shift = 58
    self.depth_min = None
    self.depth_max = None
    self.nnum = None         # the total node number

    self.batch_id = None
    self.node_depth = None
    self.child = None        # the octree node has children or not
    self.key = None          # the bits from the 58th is the node depth
    self.octree_mask = None  # used to pad zeros for non-leaf nodes

    self.edge_idx = None     # the edge index in (i, j)
    self.edge_dir = None     # the dirction of the edge

  @property
  def node_type(self):
    return self.node_depth - self.depth_min

  @property
  def edge_type(self):
    return self.edge_dir


class OctreeD(Octree):

  def __init__(self, octree: Octree, max_depth: Optional[int] = None, **kwargs):
    super().__init__(octree.depth, octree.full_depth)
    self.__dict__.update(octree.__dict__)

    # node numbers for the octree
    self._set_node_num()

    # for the construction of the dual octree
    self.graphs = [Graph() for _ in range(self.depth + 1)]

    # build the dual octree
    self._build_lookup_tables()
    self.build_dual_graph(max_depth)

  def _set_node_num(self):
    self.ncum = ocnn.utils.cumsum(self.nnum, dim=0, exclusive=True)
    self.lnum = self.nnum - self.nnum_nempty  # leaf node numbers
    self.lnum[self.depth] = self.nnum[self.depth]

  def _build_lookup_tables(self):
    self.ngh = torch.tensor(
        [[0, 0, 1], [0, 0, -1],       # up, down
         [0, 1, 0], [0, -1, 0],       # right, left
         [1, 0, 0], [-1, 0, 0]],      # front, back
        dtype=torch.int16, device=self.device)
    self.dir_table = torch.tensor(
        [[1, 3, 5, 7], [0, 2, 4, 6],   # up, down
         [2, 3, 6, 7], [0, 1, 4, 5],   # right, left
         [4, 5, 6, 7], [0, 1, 2, 3]],  # front, back
        dtype=torch.int64, device=self.device)
    self.dir_type = torch.tensor(
        [0, 1, 2, 3, 4, 5],
        dtype=torch.int64, device=self.device)
    self.dir_remap = torch.tensor(
        [1, 0, 3, 2, 5, 4],
        dtype=torch.int64, device=self.device)
    self.interal_row = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
         4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7],
        dtype=torch.int64, device=self.device)
    self.interal_col = torch.tensor(
        [1, 2, 4, 0, 3, 5, 0, 3, 6, 1, 2, 7,
         0, 5, 6, 1, 4, 7, 2, 4, 7, 3, 5, 6],
        dtype=torch.int64, device=self.device)
    self.interal_dir = torch.tensor(
        [0, 2, 4, 1, 2, 4, 3, 0, 4, 3, 1, 4,
         5, 0, 2, 5, 1, 2, 5, 3, 0, 5, 3, 1],
        dtype=torch.int64, device=self.device)

  def batch_id(self, depth: int, nempty: bool = False):
    r""" Override the method in the Octree class. """
    return self.graphs[depth].batch_id

  def arange(self, num: int):
    return torch.arange(num, dtype=torch.int64, device=self.device)

  def empty_tensor(self):
    return torch.tensor([], dtype=torch.int64, device=self.device)

  def build_dual_graph(self, max_depth: Optional[int] = None):
    depth = max_depth or self.depth
    self.dense_graph(self.full_depth)
    for d in range(self.full_depth + 1, depth + 1):
      self.sparse_graph(d)

  def octree_split(self, split: torch.Tensor, depth: int):
    #  update self.children[depth], self.nnum_nempty[depth]
    super().octree_split(split, depth)

    # update the properties of the dual octree
    self._set_node_num()
    self.graphs[depth].child[-self.nnum[depth]:] = self.children[depth]

  def octree_grow(self, depth: int, update_neigh: bool = True):
    super().octree_grow(depth, update_neigh)
    if depth > self.full_depth:
      self.sparse_graph(depth)
    else:
      self.dense_graph(depth)

  def dense_graph(self, depth: int = 3):
    K = 6  # each node has at most K neighboring node
    bnd = 2 ** depth
    num = bnd ** 3

    ki = self.arange(num)
    x, y, z, _ = key2xyz(ki, depth)
    xi = torch.stack([x, y, z], dim=1)  # [N, 3]
    xj = xi.unsqueeze(1) + self.ngh     # [N, K, 3]
    xj = xj.view(-1, 3)                 # [N*K, 3]

    # for full octree, the octree key is the index
    row = ki.unsqueeze(1).repeat(1, K).view(-1)               # [N*K]
    col = xyz2key(xj[:, 0], xj[:, 1], xj[:, 2], depth=depth)  # [N*K]
    edge_dir = self.dir_type.repeat(num)                      # [K] -> [N*K]

    # remove invalid edges
    valid = torch.logical_and(xj > -1, xj < bnd)  # out-of-bound
    valid = torch.all(valid, dim=-1).view(-1)
    row, col, edge_dir = row[valid], col[valid], edge_dir[valid]

    # deal with batches
    dis = self.arange(self.batch_size)
    dis = dis.unsqueeze(1) * num
    row = row.unsqueeze(0) + dis
    col = col.unsqueeze(0) + dis
    edge_dir = edge_dir.repeat(self.batch_size)

    # graph edges
    graph = self.graphs[depth]
    graph.edge_dir = edge_dir.view(-1)
    graph.edge_idx = torch.stack([row.view(-1), col.view(-1)])

    # graph nodes
    node_key = self.keys[depth]
    graph.batch_id = node_key >> 48
    graph.node_depth = torch.ones_like(node_key) * depth
    graph.key = torch.bitwise_or(node_key, graph.node_depth << graph.key_shift)
    graph.child = self.children[depth]
    ones = torch.ones(self.nnum[depth], dtype=bool, device=self.device)
    graph.octree_mask = ones

    # depth and node numbers
    graph.depth_max = depth
    graph.depth_min = depth
    graph.nnum = node_key.numel()

  def sparse_graph(self, depth: int):
    # get the previous graph
    graph = self.graphs[depth - 1]
    edge_idx = graph.edge_idx
    edge_dir = graph.edge_dir
    leaf_mask = graph.child < 0
    idx_bias = graph.nnum

    # mark invalid nodes of layer (depth-1)
    row, col = edge_idx[0], edge_idx[1]
    valid_row = graph.child[row] < 0
    valid_col = graph.child[col] < 0
    invalid_row = torch.logical_not(valid_row)
    invalid_col = torch.logical_not(valid_col)
    valid_edges = torch.logical_and(valid_row, valid_col)
    invalid_row_vtx = torch.logical_and(invalid_row, valid_col)
    invalid_both_vtx = torch.logical_and(invalid_row, invalid_col)

    # deal with edges with invalid row vtx only
    vi = row[invalid_row_vtx]
    vj = col[invalid_row_vtx]
    di = edge_dir[invalid_row_vtx]
    row_o1 = graph.child[vi].unsqueeze(1) * 8 + self.dir_table[di, :]
    row_o1 = row_o1.view(-1) + idx_bias  # NOTE: add graph.nnum
    col_o1 = vj.unsqueeze(1).repeat(1, 4).view(-1)
    dir_o1 = di.unsqueeze(1).repeat(1, 4).view(-1)

    # deal with edges with 2 invalid nodes
    if invalid_both_vtx.any():
      vi = row[invalid_both_vtx]
      vj = col[invalid_both_vtx]
      di = edge_dir[invalid_both_vtx]
      dj = self.dir_remap[di]
      row_o2 = graph.child[vi].unsqueeze(1) * 8 + self.dir_table[di, :]
      row_o2 = row_o2.view(-1) + idx_bias  # NOTE: add graph.nnum
      col_o2 = graph.child[vj].unsqueeze(1) * 8 + self.dir_table[dj, :]
      col_o2 = col_o2.view(-1) + idx_bias  # NOTE: add graph.nnum
      dir_o2 = di.unsqueeze(1).repeat(1, 4).view(-1)
    else:
      row_o2 = self.empty_tensor()
      col_o2 = self.empty_tensor()
      dir_o2 = self.empty_tensor()

    # add internal edges connecting sibling nodes.
    row_i, col_i, dir_i = self._internal_edges(depth, idx_bias)

    # update graph edges
    graph_d = self.graphs[depth]
    graph_d.edge_idx = torch.stack([
        torch.cat([row[valid_edges], row_i, row_o1, col_o1, row_o2]),
        torch.cat([col[valid_edges], col_i, col_o1, row_o1, col_o2])])
    graph_d.edge_dir = torch.cat([
        edge_dir[valid_edges], dir_i, dir_o1, self.dir_remap[dir_o1], dir_o2])

    # update graph nodes
    node_key = self.keys[depth]
    batch_id = node_key >> 48
    node_depth = torch.ones_like(node_key) * depth
    node_key = torch.bitwise_or(node_key, node_depth << 58)
    leaf_mask = graph.child < 0
    graph_d.batch_id = torch.cat([graph.batch_id[leaf_mask], batch_id])
    graph_d.node_depth = torch.cat([graph.node_depth[leaf_mask], node_depth])
    graph_d.key = torch.cat([graph.key[leaf_mask], node_key])
    graph_d.child = torch.cat([graph.child[leaf_mask], self.children[depth]])
    octree_mask = torch.cat(self.children[graph.depth_min:depth]) < 0
    ones = torch.ones(self.nnum[depth], dtype=bool, device=self.device)
    graph_d.octree_mask = torch.cat([octree_mask, ones])

    # remap the edge index since some nodes are removed
    mask = torch.cat([leaf_mask, ones], dim=0)
    remapper = torch.cumsum(mask.long(), dim=0) - 1
    graph_d.edge_idx = remapper[graph_d.edge_idx]

    # depth and node numbers
    graph_d.depth_max = depth
    graph_d.depth_min = graph.depth_min
    graph_d.nnum = graph_d.key.numel()

  def _internal_edges(self, depth: int, idx_bias: int):
    num = int(self.nnum[depth] / 8)
    dis = self.arange(num).unsqueeze(1) * 8 + idx_bias
    row = self.interal_row.unsqueeze(0) + dis
    col = self.interal_col.unsqueeze(0) + dis
    edge_dir = self.interal_dir.repeat(num)
    return row.view(-1), col.view(-1), edge_dir.view(-1)

  def get_input_feature(self, all_leaf_nodes=True, feature='L'):
    # the initial feature of leaf nodes in the layer depth
    # data = ocnn.modules.InputFeature(feature, nempty=False)(self)
    data = super().get_input_feature(feature)

    # # to be consistent with the original code. TODO: remove this
    # flag = self.nempty_mask(self.depth).float().unsqueeze(1)
    # data = torch.cat([data, flag * (2 / 3 ** 0.5), flag], dim=1)

    # concat zero features with the initial features in layer depth
    if all_leaf_nodes:
      leaf_num = torch.sum(self.lnum[self.full_depth:-1])
      zeros = torch.zeros(leaf_num, data.shape[1], device=self.device)
      data = torch.cat([zeros, data], dim=0)

    return data
