# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import numpy as np
from ocnn.octree import Octree, Points
from thsolver import Dataset

from .utils import collate_func


class Transform:

  def __init__(self, flags):
    self.flags = flags
    self.depth = flags.depth
    self.full_depth = flags.full_depth
    self.point_sample_num = flags.point_sample_num
    self.point_scale = flags.point_scale

  def build_octree(self, points):
    pts, normals = points[:, :3], points[:, 3:]
    points_cloud = Points(pts, normals)
    octree = Octree(self.depth, self.full_depth)
    octree.build_octree(points_cloud)
    return {'octree_in': octree, 'points_in': points, 'octree_gt': octree}

  def sample_on_surface(self, points):
    '''Randomly sample points on the surface.'''

    rnd_idx = torch.randint(high=points.shape[0], size=(self.point_sample_num,))
    pos = points[rnd_idx, :3]
    normal = points[rnd_idx, 3:]
    sdf = torch.zeros(self.point_sample_num)
    return {'pos': pos, 'grad': normal, 'sdf': sdf}

  def sample_off_surface(self, bbox):
    '''Randomly sample points in the 3D space.'''

    # uniformly sampling in the whole 3D sapce
    pos = torch.rand(self.point_sample_num, 3) * 2 - 1

    # point gradients
    # grad = torch.zeros(self.point_sample_num, 3)
    norm = torch.sqrt(torch.sum(pos**2, dim=1, keepdim=True)) + 1e-6
    grad = pos / norm  # !!! fake off-surface gradients

    # sdf values
    esp = 0.04
    bbmin, bbmax = bbox[:3] - esp, bbox[3:] + esp
    mask = torch.logical_and(pos > bbmin, pos < bbmax).all(1)  # inbox
    sdf = -1.0 * torch.ones(self.point_sample_num)  # !!! fake off-surface sdf
    sdf[mask.logical_not()] = 1.0  # exactly out-of-bbox
    return {'pos': pos, 'grad': grad, 'sdf': sdf}

  def scale_and_clip_points(self, points):
    points[:, :3] = points[:, :3] * self.point_scale  # rescale points
    pts = points[:, :3]
    mask = torch.logical_and(pts > -1.0, pts < 1.0).all(1)
    return points[mask]

  def compute_bbox(self, points):
    pts = points[:, :3]
    bbmin = pts.min(0)[0] - 0.06
    bbmax = pts.max(0)[0] + 0.06
    bbmin = torch.clamp(bbmin, min=-1, max=1)
    bbmax = torch.clamp(bbmax, min=-1, max=1)
    return torch.cat([bbmin, bbmax])

  def __call__(self, points, idx):
    points = self.scale_and_clip_points(points)  # clip points to [-1, 1]
    output = self.build_octree(points)
    bbox = self.compute_bbox(points)
    output['bbox'] = bbox  # used in marching cubes

    sdf_on_surf = self.sample_on_surface(points)
    sdf_off_surf = self.sample_off_surface(bbox)
    sdfs = {key: torch.cat([sdf_on_surf[key], sdf_off_surf[key]], dim=0)
            for key in sdf_on_surf.keys()}
    output.update(sdfs)
    return output


class SingleTransform(Transform):

  def __init__(self, flags):
    super().__init__(flags)
    self.output = None
    self.points = None
    self.bbox = None

  def __call__(self, points, idx):
    if self.output is None:
      self.points = self.scale_and_clip_points(points)  # clip points to [-1, 1]
      self.output = self.build_octree(self.points)
      self.bbox = self.compute_bbox(self.points)
      self.output['bbox'] = self.bbox  # used in marching cubes

    output = self.output
    sdf_on_surf = self.sample_on_surface(self.points)
    sdf_off_surf = self.sample_off_surface(self.bbox)
    sdfs = {key: torch.cat([sdf_on_surf[key], sdf_off_surf[key]], dim=0)
            for key in sdf_on_surf.keys()}
    output.update(sdfs)
    return output


def read_file(filename: str):
  if filename.endswith('.xyz'):
    points = np.loadtxt(filename)
  elif filename.endswith('.npy'):
    points = np.load(filename)
  else:
    raise NotImplementedError
  output = torch.from_numpy(points.astype(np.float32))
  return output


def get_pointcloud_dataset(flags):
  transform = Transform(flags)
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_file, in_memory=flags.in_memory)
  return dataset, collate_func


def get_singlepointcloud_dataset(flags):
  transform = SingleTransform(flags)
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_file, in_memory=flags.in_memory)
  return dataset, collate_func
