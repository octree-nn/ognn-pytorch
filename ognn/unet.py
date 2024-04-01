# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch

from ognn.octreed import OctreeD
from ognn.ounet import GraphOUNet


class GraphUNet(GraphOUNet):

  def __init__(self, in_channels: int, resblk_type: str = 'basic',
               feature: str = 'L', norm_type: str = 'batch_norm',
               act_type: str = 'relu', **kwargs):
    super().__init__(
        in_channels, resblk_type, feature, norm_type, act_type)
    self.predict = None

  def config_network(self):
    self.n_node_type = 7
    self.encoder_blk_nums = [2, 2, 2, 2, 2, 2]
    self.decoder_blk_nums = [2, 2, 2, 2, 2, 2]
    self.encoder_channels = [32, 32, 32, 64, 128, 256]
    self.decoder_channels = [256, 128, 64, 32, 32, 32]

  def octree_decoder(self, convs: dict, octree: OctreeD):
    signals = dict()
    depth = octree.depth - self.encoder_stages + 1
    deconv = convs[depth]
    for i in range(self.decoder_stages):
      d = depth + i
      if i > 0:
        deconv = self.upsample[i-1](deconv, octree, d-1)
        deconv = deconv + convs[d]  # skip connections
        deconv = self.decoder[i-1](deconv, octree, d)

      # regress signals and pad zeros to non-leaf nodes
      signal = self.regress[i](deconv, octree, d)
      signals[d] = self.graph_pad(signal, octree, d)

    return {'signals': signals}

  def forward(self, octree: OctreeD, octree_out: OctreeD,
              pos: torch.Tensor = None, update_octree: bool = False):
    # run encoder and decoder
    convs = self.octree_encoder(octree)
    output = self.octree_decoder(convs, octree)

    # setup mpu
    depth_out = octree.depth
    self.neural_mpu.setup(output['signals'], octree, depth_out)

    # compute function value with mpu
    if pos is not None:
      output['mpus'] = self.neural_mpu(pos)

    # create the mpu wrapper
    output['neural_mpu'] = lambda pos: self.neural_mpu(pos)[depth_out]
    return output
