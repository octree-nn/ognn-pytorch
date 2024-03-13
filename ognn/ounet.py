# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch
import torch.nn

from ocnn.octree import Octree
from ognn.octreed import OctreeD
from ognn import mpu, nn
from easydict import EasyDict as edict


class GraphOUNet(torch.nn.Module):

  def __init__(self, in_channels: int):
    super().__init__()
    self.bottleneck = 4
    self.n_edge_type = 7
    self.n_node_type = 5
    self.in_channels = in_channels
    self.config_network()

    self.neural_mpu = mpu.NeuralMPU()
    self.graph_pad = nn.GraphPad()
    self.encoder_stages = len(self.encoder_blk_nums)
    self.decoder_stages = len(self.decoder_blk_nums)

    # encoder
    n_node_types = [self.n_node_type - i for i in range(self.encoder_stages)]
    self.conv1 = nn.GraphConvNormRelu(
        in_channels, self.encoder_channels[0], self.n_edge_type,
        n_node_types[0], self.group, self.norm_type)
    self.encoder_blks = torch.nn.ModuleList([nn.GraphResBlocks(
        self.encoder_channels[i], self.encoder_channels[i],
        self.n_edge_type, n_node_types[i], self.group, self.norm_type,
        self.bottleneck, self.encoder_blk_nums[i], self.resblk_type)
        for i in range(self.encoder_stages)])
    self.downsample = torch.nn.ModuleList([nn.GraphDownsample(
        self.encoder_channels[i], self.encoder_channels[i+1], self.group,
        self.norm_type) for i in range(self.encoder_stages-1)])

    # decoder
    current_n_node_type = self.n_node_type - self.encoder_stages
    n_node_types = [current_n_node_type + i for i in range(self.decoder_stages)]
    self.upsample = torch.nn.ModuleList([nn.GraphUpsample(
        self.decoder_channels[i], self.decoder_channels[i+1], self.group,
        self.norm_type) for i in range(self.decoder_stages)])
    self.decoder_blks = torch.nn.ModuleList([nn.GraphResBlocks(
        self.decoder_channels[i+1], self.decoder_channels[i+1],
        self.n_edge_type, n_node_types[i], self.group, self.norm_type,
        self.bottleneck, self.decoder_blk_nums[i], self.resblk_type)
        for i in range(self.decoder_stages)])

    # header
    self.predict = torch.nn.ModuleList([
         self._make_predict_module(self.decoder_channels[i], 2)
         for i in range(self.decoder_stages)])
    self.regress = torch.nn.ModuleList([
         self._make_predict_module(self.decoder_channels[i], 4)
         for i in range(self.decoder_stages)])

  def config_network(self):
    self.head_channel = 64
    self.group = 1    # for group normalization
    self.norm_type = 'batch_norm'
    self.resblk_type = 'bottleneck'
    self.encoder_blk_nums = [3, 3, 3, 3]
    self.decoder_blk_nums = [3, 3, 3, 3]
    self.encoder_channels = [32, 32, 64, 128, 256]
    self.decoder_channels = [256, 128, 64, 32, 32]

  def _make_predict_module(self, in_channels: int, out_channels: int = 2,
                           num_hidden: int = 32):
    return torch.nn.Sequential(
        nn.Conv1x1NormRelu(in_channels, num_hidden, self.group, self.norm_type),
        nn.Conv1x1(num_hidden, out_channels, use_bias=True))

  def _octree_align(self, value: torch.Tensor, octree: OctreeD,
                    octree_query: OctreeD, depth: int):
    key = octree.graphs[depth].key
    query = octree_query.graphs[depth].key
    assert key.shape[0] == value.shape[0]
    return ocnn.nn.search_value(value, key, query)

  def encoder(self, octree: OctreeD, depth: int):
    convs = dict()  # graph convolution features
    data = octree.get_input_feature(feature='L')
    convs[depth] = self.conv1(data, octree, depth)
    for i in range(self.encoder_stages):
      d = depth - i
      convs[d] = self.encoder_blks[i](convs[d], octree, d)
      if i < self.encoder_stages - 1:
        convs[d-1] = self.downsample[i](convs[d], octree, d)
    return convs

  def decoder(self, convs: dict, octree_in: OctreeD, octree_out: OctreeD,
              depth_start: int, depth_end: int, update_octree: bool = False):
    assert depth_end - depth_start + 1 == self.decoder_stages

    logits, signals = dict(), dict()
    deconv = convs[depth_start]
    for i in range(self.decoder_stages):
      d = depth_start + i
      deconv = self.decoder_blks[i](deconv, octree_out, d)

      # predict the splitting label and signal
      logit = self.predict[i](deconv)
      nnum = octree_out.nnum[d]
      logits[d] = logit[-nnum:]

      # regress signals and pad zeros to non-leaf nodes
      signal = self.regress[i](deconv)
      signals[d] = self.graph_pad(signal, octree_out, d)

      if i < self.decoder_stages - 1:  # skip the last stage
        deconv = self.upsample[i](deconv, octree_out, d)
        skip = self._octree_align(convs[d], octree_in, octree_out, d)
        deconv = deconv + skip  # output-guided skip connections

      # update the octree according to predicted labels
      if update_octree:
        split = logits[d].argmax(1).int()
        octree_out.octree_split(split, d)
        if d < depth_end:
          octree_out.octree_grow(d + 1)

    return edict({'logits': logits, 'signals': signals})

  def create_full_octree(self, depth: int, octree: Octree):
    octree = Octree(depth, octree.full_depth, octree.batch_size, octree.device)
    for d in range(octree.full_depth+1):
      octree.octree_grow_full(d)
    octree_out = OctreeD(octree)
    octree_out.build_dual_graph()
    return octree_out

  def forward(self, octree_in, octree_out, depth_out, pos = None):
    # generate dual octrees
    octree_in = OctreeD(octree_in)
    octree_in.build_dual_graph()

    # initialize the output octree
    update_octree = octree_out is None
    if update_octree:
      octree_out = self.create_full_octree(depth_out, octree_in)

    # run encoder and decoder
    convs = self.encoder(octree_in, octree_in.depth)
    output = self.decoder(
        convs, octree_in, octree_out, depth_out, update_octree)

    # create the mpu wrapper
    def _neural_mpu(pos):
      pred = self.neural_mpu(pos, output.signal, octree_out, depth_out)
      return pred[self.depth_out][0]
    output['neural_mpu'] = _neural_mpu

    # compute function value with mpu
    if pos is not None:
      output['mpus'] = _neural_mpu(pos)

    return output
