# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch

from ognn import nn
from ognn.ounet import GraphOUNet
from ognn.octreed import OctreeD


class GraphAE(GraphOUNet):

  def __init__(self, in_channels: int, resblk_type: str = 'basic',
               code_channel: int = 8):
    super().__init__(in_channels, resblk_type)

    # reduce the code channel for fair comparison; if the code is of high
    # dimension, the performance will be significantly better
    self.code_channel = code_channel
    # code_dim = self.code_channel * 2 ** (3 * full_depth)

    self.project1 = nn.Conv1x1NormRelu(
        self.encoder_channels[-1], self.code_channel)
    self.project2 = nn.Conv1x1NormRelu(
        self.code_channel, self.decoder_channels[0])

  def octree_encoder(self, octree: OctreeD):
    convs = super().octree_encoder(octree)
    depth = octree.depth - self.encoder_stages + 1

    # reduce the dimension to get the shape code
    shape_code = self.project1(convs[depth])
    return shape_code

  def octree_decoder(self, shape_code, octree_in: OctreeD, octree_out: OctreeD,
                     update_octree: bool = False):
    logits, signals = dict(), dict()
    depth = octree_in.depth - self.encoder_stages + 1
    deconv = self.project2(shape_code)
    for i in range(self.decoder_stages):
      d = depth + i
      if i > 0:
        deconv = self.upsample[i-1](deconv, octree_out, d-1)
        # there is no skip connections between the encoder and decoder
        deconv = self.decoder[i-1](deconv, octree_out, d)

      # predict the splitting label and signal
      logit = self.predict[i](deconv)
      nnum = octree_out.nnum[d]
      logits[d] = logit[-nnum:]

      # regress signals and pad zeros to non-leaf nodes
      signal = self.regress[i](deconv)
      signals[d] = self.graph_pad(signal, octree_out, d)

      # update the octree according to predicted labels
      if update_octree:
        split = logits[d].argmax(1).int()
        octree_out.octree_split(split, d)
        if i < self.decoder_stages - 1:
          octree_out.octree_grow(d + 1)

    return {'logits': logits, 'signals': signals, 'octree_out': octree_out}
