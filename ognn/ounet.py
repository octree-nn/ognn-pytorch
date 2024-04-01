# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch
import torch.nn

from ognn import mpu, nn
from ognn.octreed import OctreeD


class GraphOUNet(torch.nn.Module):

  def __init__(self, in_channels: int, resblk_type: str = 'basic',
               feature: str = 'L', norm_type: str = 'batch_norm',
               act_type: str = 'relu', **kwargs):
    super().__init__()
    self.bottleneck = 4
    self.n_edge_type = 7
    self.head_channel = 64
    self.in_channels = in_channels
    self.resblk_type = resblk_type
    self.feature = feature
    self.norm_type = norm_type
    self.act_type = act_type
    self.config_network()

    self.neural_mpu = mpu.NeuralMPU()
    self.graph_pad = nn.GraphPad()
    self.encoder_stages = len(self.encoder_blk_nums)
    self.decoder_stages = len(self.decoder_blk_nums)

    # encoder
    n_node_types = [self.n_node_type - i for i in range(self.encoder_stages)]
    self.conv1 = nn.GraphConvNormAct(
        in_channels, self.encoder_channels[0], self.n_edge_type,
        n_node_types[0], self.norm_type, self.act_type)
    self.encoder = torch.nn.ModuleList([nn.GraphResBlocks(
        self.encoder_channels[i], self.encoder_channels[i],
        self.n_edge_type, n_node_types[i], self.norm_type,
        self.act_type, self.bottleneck, self.encoder_blk_nums[i],
        self.resblk_type) for i in range(self.encoder_stages)])
    self.downsample = torch.nn.ModuleList([nn.GraphDownsample(
        self.encoder_channels[i], self.encoder_channels[i+1],
        self.norm_type, self.act_type) for i in range(self.encoder_stages - 1)])

    # decoder
    n_node_type = self.n_node_type - self.encoder_stages + 1
    n_node_types = [n_node_type + i for i in range(self.decoder_stages)]
    self.upsample = torch.nn.ModuleList([nn.GraphUpsample(
        self.decoder_channels[i - 1], self.decoder_channels[i],
        self.norm_type, self.act_type) for i in range(1, self.decoder_stages)])
    self.decoder = torch.nn.ModuleList([nn.GraphResBlocks(
        self.decoder_channels[i], self.decoder_channels[i],
        self.n_edge_type, n_node_types[i], self.norm_type,
        self.act_type, self.bottleneck, self.decoder_blk_nums[i],
        self.resblk_type) for i in range(1, self.decoder_stages)])

    # header
    mid_channels = 32
    self.predict = torch.nn.ModuleList([nn.Prediction(
        self.decoder_channels[i], mid_channels, 2, self.norm_type,
        self.act_type) for i in range(self.decoder_stages)])
    self.regress = torch.nn.ModuleList([nn.Prediction(
        self.decoder_channels[i], mid_channels, 4, self.norm_type,
        self.act_type) for i in range(self.decoder_stages)])

  def config_network(self):
    self.n_node_type = 5
    self.encoder_blk_nums = [3, 3, 3, 3]
    self.decoder_blk_nums = [3, 3, 3, 3]
    self.encoder_channels = [32, 64, 128, 256]
    self.decoder_channels = [256, 128, 64, 32]

  def _octree_align(self, value: torch.Tensor, octree: OctreeD,
                    octree_query: OctreeD, depth: int):
    key = octree.graphs[depth].key
    query = octree_query.graphs[depth].key
    assert key.shape[0] == value.shape[0]
    return ocnn.nn.search_value(value, key, query)

  def octree_encoder(self, octree: OctreeD):
    convs = dict()  # graph convolution features
    depth = octree.depth
    data = octree.get_input_feature(feature=self.feature)
    convs[depth] = self.conv1(data, octree, depth)
    for i in range(self.encoder_stages):
      d = depth - i
      convs[d] = self.encoder[i](convs[d], octree, d)
      if i < self.encoder_stages - 1:
        convs[d-1] = self.downsample[i](convs[d], octree, d)
    return convs

  def octree_decoder(self, convs: dict, octree_in: OctreeD, octree_out: OctreeD,
                     update_octree: bool = False):
    logits, signals = dict(), dict()
    depth = octree_in.depth - self.encoder_stages + 1
    deconv = convs[depth]
    for i in range(self.decoder_stages):
      d = depth + i
      if i > 0:
        deconv = self.upsample[i-1](deconv, octree_out, d-1)
        skip = self._octree_align(convs[d], octree_in, octree_out, d)
        deconv = deconv + skip  # output-guided skip connections
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

  def forward(self, octree_in: OctreeD, octree_out: OctreeD,
              pos: torch.Tensor = None, update_octree: bool = False):
    # run encoder and decoder
    convs = self.octree_encoder(octree_in)
    output = self.octree_decoder(convs, octree_in, octree_out, update_octree)

    # setup mpu
    depth_out = octree_out.depth
    self.neural_mpu.setup(output['signals'], octree_out, depth_out)

    # compute function value with mpu
    if pos is not None:
      output['mpus'] = self.neural_mpu(pos)

    # create the mpu wrapper
    output['neural_mpu'] = lambda pos: self.neural_mpu(pos)[depth_out]
    return output
