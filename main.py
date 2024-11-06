# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import ocnn

from thsolver import Solver, get_config
from ognn.octreed import OctreeD

import builder
import utils


class OGNSolver(Solver):

  def get_model(self, flags):
    return builder.get_model(flags)

  def get_dataset(self, flags):
    return builder.get_dataset(flags)

  def batch_to_cuda(self, batch):
    keys = [
        'octree_in', 'octree_gt', 'pos', 'sdf',
        'grad', 'weight', 'occu', 'color']
    for key in keys:
      if key in batch:
        batch[key] = batch[key].cuda()
    batch['pos'].requires_grad_()

  def compute_loss(self, batch, model_out):
    flags = self.FLAGS.LOSS
    loss_func = builder.get_loss_function(flags)
    output = loss_func(batch, model_out, flags.loss_type, **flags)
    return output

  def model_forward(self, batch):
    self.batch_to_cuda(batch)
    octree_in = OctreeD(batch['octree_in'])
    octree_gt = OctreeD(batch['octree_gt'])
    model_out = self.model(octree_in, octree_gt, batch['pos'])

    output = self.compute_loss(batch, model_out)
    losses = [val for key, val in output.items() if 'loss' in key]
    output['loss'] = torch.sum(torch.stack(losses))
    return output

  def train_step(self, batch):
    output = self.model_forward(batch)
    output = {'train/' + key: val for key, val in output.items()}
    return output

  def test_step(self, batch):
    output = self.model_forward(batch)
    output = {'test/' + key: val for key, val in output.items()}
    return output

  def eval_step(self, batch):
    # forward the model
    octree_in = OctreeD(batch['octree_in'].cuda())
    octree_out = self._init_octree_out(octree_in)
    output = self.model.forward(octree_in, octree_out, update_octree=True)

    # extract the mesh
    flags = self.FLAGS.DATA.test
    filename = self._extract_filename(batch)
    bbmin, bbmax = self._get_bbox(batch)
    utils.create_mesh(
        output['neural_mpu'], filename, size=flags.resolution,
        bbmin=bbmin, bbmax=bbmax, mesh_scale=flags.point_scale,
        save_sdf=flags.save_sdf)

    # save the input point cloud
    filename = filename[:-4] + '.input.ply'
    points = batch['points_in'][0]
    points.points *= flags.point_scale
    utils.points2ply(filename, points)

  def _extract_filename(self, batch):
    filename = batch['filename'][0]
    pos = filename.rfind('.')
    if pos != -1: filename = filename[:pos]  # remove the suffix
    filename = os.path.join(self.logdir, filename + '.obj')
    folder = os.path.dirname(filename)
    if not os.path.exists(folder): os.makedirs(folder)
    return filename

  def _init_octree_out(self, octree_in):
    full_depth = octree_in.full_depth  # grow octree to full_depth
    octree_out = ocnn.octree.init_octree(
        full_depth, full_depth, octree_in.batch_size, octree_in.device)
    return OctreeD(octree_out, full_depth)

  def _get_bbox(self, batch):
    if 'bbox' in batch:
      bbox = batch['bbox'][0].numpy()
      bbmin, bbmax = bbox[:3], bbox[3:]
    else:
      sdf_scale = self.FLAGS.DATA.test.sdf_scale
      bbmin, bbmax = -sdf_scale, sdf_scale
    return bbmin, bbmax

  @classmethod
  def update_configs(cls):
    FLAGS = get_config()
    # Additional default and common configurations
    FLAGS.SOLVER.with_color = False           # extract colored meshes
    FLAGS.SOLVER.save_sdf = False             # save the sdf values


if __name__ == '__main__':
  OGNSolver.main()
