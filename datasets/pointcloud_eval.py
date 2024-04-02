# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import numpy as np
from plyfile import PlyData

from thsolver import Dataset
from ocnn.octree import Octree, Points
from .utils import collate_func


class Transform:
  r'''Load point clouds from ply files, rescale the points and build octree.
  Used to evaluate the network trained on ShapeNet.'''

  def __init__(self, flags):
    self.flags = flags
    self.depth = flags.depth
    self.full_depth = flags.full_depth
    self.point_scale = flags.point_scale

  def points2octree(self, points: Points):
    octree = Octree(self.depth, self.full_depth)
    octree.build_octree(points)
    return octree

  def __call__(self, points, idx):
    # After normalization, the points are in [-1, 1]
    points = points[:, :3] / self.point_scale

    # construct the points and octree
    points_in = Points(points=torch.from_numpy(points).float(),
                       features=torch.ones(points.shape[0], 1).float())
    points_in.clip(-1.0, 1.0)
    octree_in = self.points2octree(points_in)

    return {'points_in': points, 'octree_in': octree_in}


def read_file(filename: str):
  plydata = PlyData.read(filename + '.ply')
  vtx = plydata['vertex']
  points = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=1).astype(np.float32)
  return points


def get_pointcloud_eval_dataset(flags):
  transform = Transform(flags)
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_file, in_memory=flags.in_memory)
  return dataset, collate_func
