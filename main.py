# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import ocnn

from thsolver import Solver
from ognn.octreed import OctreeD

import builder
import utils


class OGNSolver(Solver):

  def get_model(self, flags):
    return builder.get_model(flags)

  def get_dataset(self, flags):
    return builder.get_dataset(flags)

  def batch_to_cuda(self, batch):
    keys = ['octree_in', 'octree_gt', 'pos', 'sdf', 'grad', 'weight', 'occu']
    for key in keys:
      if key in batch:
        batch[key] = batch[key].cuda()
    batch['pos'].requires_grad_()

  def compute_loss(self, batch, model_out):
    flags = self.FLAGS.LOSS
    loss_func = builder.get_loss_function(flags)
    output = loss_func(batch, model_out, flags.loss_type)
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
    depth_out = self.FLAGS.MODEL.depth_out
    octree_in = OctreeD(batch['octree_in'].cuda())
    octree_out = self._init_octree_out(octree_in, depth_out)
    output = self.model.forward(octree_in, octree_out, update_octree=True)

    # extract the mesh
    filename = batch['filename'][0]
    pos = filename.rfind('.')
    if pos != -1: filename = filename[:pos]  # remove the suffix
    filename = os.path.join(self.logdir, filename + '.obj')
    folder = os.path.dirname(filename)
    if not os.path.exists(folder): os.makedirs(folder)
    bbmin, bbmax = self._get_bbox(batch)
    utils.create_mesh(
        output['neural_mpu'], filename, size=self.FLAGS.SOLVER.resolution,
        bbmin=bbmin, bbmax=bbmax, mesh_scale=self.FLAGS.DATA.test.point_scale,
        save_sdf=self.FLAGS.SOLVER.save_sdf)

    # save the input point cloud
    filename = filename[:-4] + '.input.ply'
    points = batch['points_in'][0]
    points[:, :3] *= self.FLAGS.DATA.test.point_scale
    utils.points2ply(filename, batch['points_in'][0])

  def _init_octree_out(self, octree_in, depth_out):
    full_depth = octree_in.full_depth  # grow octree to full_depth
    octree_out = ocnn.octree.init_octree(
        depth_out, full_depth, octree_in.batch_size, octree_in.device)
    return OctreeD(octree_out, full_depth)

  def _get_bbox(self, batch):
    if 'bbox' in batch:
      bbox = batch['bbox'][0].numpy()
      bbmin, bbmax = bbox[:3], bbox[3:]
    else:
      sdf_scale = self.FLAGS.SOLVER.sdf_scale
      bbmin, bbmax = -sdf_scale, sdf_scale
    return bbmin, bbmax


if __name__ == '__main__':
  OGNSolver.main()
