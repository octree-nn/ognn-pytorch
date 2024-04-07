# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import numpy as np

from thsolver import Dataset
from ocnn.octree import Octree, Points
from .utils import collate_func


class TransformShape:

  def __init__(self, flags):
    self.flags = flags

    self.sdf_sample_num = flags.sdf_sample_num
    self.color_sample_num = flags.color_sample_num
    self.points_scale = 0.5  # the points are in [-0.5, 0.5]

    self.depth = flags.depth
    self.full_depth = flags.full_depth

  def points2octree(self, points: Points):
    octree = Octree(self.depth, self.full_depth)
    octree.build_octree(points)
    return octree

  def process_points_cloud(self, sample):
    # get the input
    points = torch.from_numpy(sample['points']).float()
    normals = torch.from_numpy(sample['normals']).float()
    colors = (torch.from_numpy(sample['colors']).float()
              if 'colors' in sample else None)
    points = points / self.points_scale  # scale to [-1.0, 1.0]

    # transform points to octree
    points_gt = Points(points=points, normals=normals, features=colors)
    points_gt.clip(min=-1, max=1)
    octree_gt = self.points2octree(points_gt)

    # construct the output dict
    return {'octree_in': octree_gt, 'octree_gt': octree_gt}

  def sample_sdf(self, sample):
    sdf = sample['sdf']
    grad = sample['grad']
    points = sample['points'] / self.points_scale  # to [-1, 1]

    rand_idx = np.random.choice(points.shape[0], size=self.sdf_sample_num)
    points = torch.from_numpy(points[rand_idx]).float()
    sdf = torch.from_numpy(sdf[rand_idx]).float()
    grad = torch.from_numpy(grad[rand_idx]).float()
    return {'pos': points, 'sdf': sdf, 'grad': grad}

  def sample_color(self, sample):
    color = sample['color']
    points = sample['points'] / self.points_scale  # to [-1, 1]

    rand_idx = np.random.choice(points.shape[0], size=self.color_sample_num)
    points = torch.from_numpy(points[rand_idx]).float()
    color = torch.from_numpy(color[rand_idx]).float()
    return {'pos_color': points, 'color': color}

  def __call__(self, sample, idx):
    output = self.process_points_cloud(sample['point_cloud'])

    sdf_samples = self.sample_sdf(sample['sdf'])
    output.update(sdf_samples)

    if self.flags.load_color:
      color_samples = self.sample_color(sample['color'])
      output.update(color_samples)
    return output


class ReadFile:

  def __init__(self, load_color=False):
    self.load_color = load_color

  def __call__(self, filename):
    filename_pc = os.path.join(filename, 'pointcloud.npz')
    raw = np.load(filename_pc)
    point_cloud = {'points': raw['points'], 'normals': raw['normals']}

    filename_sdf = os.path.join(filename, 'sdf.npz')
    raw = np.load(filename_sdf)
    sdf = {'points': raw['points'], 'grad': raw['grad'], 'sdf': raw['sdf']}

    if self.load_color:
      filename_color = os.path.join(filename, 'color.npz')
      raw = np.load(filename_color)
      point_cloud['colors'] = raw['colors']
      color = {'points': point_cloud['points'], 'color': raw['color']}
    else:
      color = None
    return {'point_cloud': point_cloud, 'sdf': sdf, 'color': color}


def get_shapenet_vae_dataset(flags):
  transform = TransformShape(flags)
  read_file = ReadFile(flags.load_color)
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_file, in_memory=flags.in_memory)
  return dataset, collate_func
