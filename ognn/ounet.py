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
from .octreed import OctreeD
from . import mpu, nn


class GraphOUNet(torch.nn.Module):

  def __init__(self, in_channels: int):
    super().__init__()
    self.config_network()
    self.in_channels = in_channels
    self.n_edge_type = 7
    self.n_node_type = 7
    self.group = 1

    self.neural_mpu = mpu.NeuralMPU()
    self.encoder_stages = len(self.encoder_blocks)
    self.decoder_stages = len(self.decoder_blocks)

    # encoder
    self.conv1 = nn.GraphConvNormRelu(
        in_channels, self.encoder_channels[0], self.n_edge_type,
        self.n_node_type, self.group, self.norm_type)
    self.downsample = torch.nn.ModuleList([nn.GraphDownsample(
        self.encoder_channels[i], self.encoder_channels[i+1], self.group,
        self.norm_type) for i in range(self.encoder_stages)])
    self.encoder = torch.nn.ModuleList([nn.GraphResBlocks(
        self.encoder_channels[i+1], self.encoder_channels[i+1],
        self.n_edge_type, self.n_node_type, self.group, self.norm_type,
        self.bottleneck, self.encoder_blocks[i], self.resblk_type)
        for i in range(self.encoder_stages)])

    # decoder
    channels = [self.decoder_channels[i+1] + self.encoder_channels[-i-2]
                for i in range(self.decoder_stages)]
    self.upsample = torch.nn.ModuleList([nn.GraphUpsample(
        self.decoder_channels[i], self.decoder_channels[i+1], self.group,
        self.norm_type) for i in range(self.decoder_stages)])
    self.decoder = torch.nn.ModuleList([nn.GraphResBlocks(
        channels[i], self.decoder_channels[i+1],
        self.n_edge_type, self.n_node_type, self.group, self.norm_type,
        self.bottleneck, self.decoder_blocks[i], self.resblk_type)
        for i in range(self.decoder_stages)])

    # header
    self.predict = torch.nn.ModuleList(
        [self._make_predict_module(self.decoder_channels[i], 2)
         for i in range(self.decoder_stages)])
    self.regress = torch.nn.ModuleList(
        [self._make_predict_module(self.decoder_channels[i], 4)
         for i in range(self.decoder_stages)])

  # def _setup_channels_and_resblks(self):
  #   # self.resblk_num = [3] * 7 + [1] + [1] * 9
  #   self.resblk_num = [3] * 16
  #   self.channels = [4, 512, 512, 256, 128, 64, 32, 32, 24]

  def config_network(self):
    r''' Configure the network channels and Resblock numbers.
    '''

    self.encoder_channels = [32, 32, 64, 128, 256]
    self.decoder_channels = [256, 256, 128, 96, 96]
    self.encoder_blocks = [2, 3, 4, 6]
    self.decoder_blocks = [2, 2, 2, 2]
    self.head_channel = 64
    self.bottleneck = 1
    self.norm_type = 'batch_norm'
    self.resblk_type = 'bottleneck'

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

  def octree_encoder(self, octree: OctreeD, depth: int):
    convs = dict()  # graph convolution features
    data = octree.get_input_feature(feature='L')
    convs[depth] = self.conv1(data, octree, depth)
    for i in range(self.encoder_stages):
      d = depth - i
      convd = self.downsample[i](convs[d], octree, d)
      convs[d-1] = self.encoder[i](convd, octree, d-1)
    return convs

  def octree_decoder(self, convs: dict, octree_in: OctreeD, octree_out: OctreeD,
                     depth_out: int, update_octree: bool = False):

    logits = dict()
    signals = dict()
    deconvs = dict()

    full_depth = octree_in.full_depth
    deconv = convs[full_depth]
    for i, d in enumerate(range(full_depth, depth_out+1)):
      if d > self.full_depth:
        deconv = self.upsample[i-1](deconv, octree_out, d-1)
        skip = self._octree_align(convs[d], octree_in, octree_out, d)
        deconv = deconv + skip  # output-guided skip connections
      deconv = self.decoder[i](deconv, octree_out, d)

      # predict the splitting label and signal
      logit = self.predict[i](deconv)
      nnum = octree_out.nnum[d]
      logits[d] = logit[-nnum:]

      signal = self.regress[i](deconv)
      # pad zeros to reg_vox to reuse the original code for ocnn
      node_mask = octree_out.graph[d]['node_mask']
      shape = (node_mask.shape[0], signal.shape[1])
      reg_vox_pad = torch.zeros(shape, device=signal.device)
      reg_vox_pad[node_mask] = signal
      signals[d] = reg_vox_pad

      # update the octree according to predicted labels
      if update_octree:
        split = logits[d].argmax(1).int()
        octree_out.octree_split(split, d)
        if d < depth_out:
          octree_out.octree_grow(d + 1)

    return {'logits': logits, 'signals': signals, 'octree_out': octree_out}

  def create_full_octree(self, octree_in: Octree):
    device = octree_in.device
    batch_size = octree_in.batch_size
    octree = Octree(self.depth, self.full_depth, batch_size, device)
    for d in range(self.full_depth+1):
      octree.octree_grow_full(depth=d)
    # doctree_out = dual_octree.DualOctree(octree_out)
    # doctree_out.post_processing_for_docnn()
    return octree

  def forward(self, octree_in, octree_out, depth_out: int):
    # generate dual octrees
    octree_in = OctreeD(octree_in)
    octree_in.post_processing_for_docnn()

    update_octree = octree_out is None
    if update_octree:
      octree_out = self.create_full_octree(octree_in)

    # run encoder and decoder
    convs = self.octree_encoder(octree_in, octree_in.depth)
    output = self.octree_decoder(
        convs, octree_in, octree_out, depth_out, update_octree)

    return output
