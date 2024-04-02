# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

from ognn import nn
from ognn.ounet import GraphOUNet
from ognn.octreed import OctreeD


class GraphAE(GraphOUNet):

  def __init__(self, in_channels: int, resblk_type: str = 'basic',
               feature: str = 'L', norm_type: str = 'batch_norm',
               act_type: str = 'relu', code_channel: int = 8, **kwargs):
    super().__init__(
        in_channels, resblk_type, feature, norm_type, act_type)

    # reduce the code channel for fair comparison; if the code is of high
    # dimension, the performance will be significantly better
    self.code_channel = code_channel
    # code_dim = self.code_channel * 2 ** (3 * full_depth)

    self.project1 = nn.Conv1x1NormAct(
        self.encoder_channels[-1], self.code_channel,
        self.norm_type, self.act_type)
    self.project2 = nn.Conv1x1NormAct(
        self.code_channel, self.decoder_channels[0],
        self.norm_type, self.act_type)

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
      logit = self.predict[i](deconv, octree_out, d)
      nnum = octree_out.nnum[d]
      logits[d] = logit[-nnum:]

      # regress signals and pad zeros to non-leaf nodes
      signal = self.regress[i](deconv, octree_out, d)
      signals[d] = self.graph_pad(signal, octree_out, d)

      # update the octree according to predicted labels
      if update_octree:
        split = logits[d].argmax(1).int()
        octree_out.octree_split(split, d)
        if i < self.decoder_stages - 1:
          octree_out.octree_grow(d + 1)

    return {'logits': logits, 'signals': signals, 'octree_out': octree_out}
