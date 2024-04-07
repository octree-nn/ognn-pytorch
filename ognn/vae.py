# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from typing import List
from torch.utils.checkpoint import checkpoint

from ognn import nn
from ognn import mpu
from ognn.octreed import OctreeD


class TinyNet(torch.nn.Module):

  def __init__(self, channels: List[int], resblk_nums: List[int],
               resblk_type: str = 'basic', bottleneck: int = 2,
               n_node_type: int = -1, **kwargs):
    super().__init__()
    self.channels = channels
    self.resblk_nums = resblk_nums
    self.resblk_type = resblk_type
    self.bottleneck = bottleneck
    self.n_node_type = n_node_type

    self.stage_num = len(self.channels)
    self.n_edge_type = 7
    self.norm_type = 'group_norm'
    self.act_type = 'gelu'
    self.use_checkpoint = True

    # encoder
    n_node_types = [self.n_node_type - i for i in range(self.stage_num)]
    self.encoder = torch.nn.ModuleList([nn.GraphResBlocks(
        self.channels[i], self.channels[i], self.n_edge_type,
        n_node_types[i], self.norm_type, self.act_type, self.bottleneck,
        self.resblk_nums[i], self.resblk_type, self.use_checkpoint)
        for i in range(self.stage_num)])
    self.downsample = torch.nn.ModuleList([nn.GraphDownsample(
        self.channels[i], self.channels[i+1], self.norm_type, self.act_type)
        for i in range(self.stage_num - 1)])

    # decoder
    self.decoder = torch.nn.ModuleList([nn.GraphResBlocks(
        self.channels[i], self.channels[i], self.n_edge_type,
        n_node_types[i], self.norm_type, self.act_type, self.bottleneck,
        self.resblk_nums[i], self.resblk_type, self.use_checkpoint)
        for i in range(self.stage_num-1, -1, -1)])
    self.upsample = torch.nn.ModuleList([nn.GraphUpsample(
        self.channels[i], self.channels[i-1], self.norm_type, self.act_type)
        for i in range(self.stage_num-1, 0, -1)])

  def forward(self, data: torch.Tensor, octree: OctreeD, depth: int):

    convs = dict()
    convs[depth] = data
    for i in range(self.stage_num):
      d = depth - i
      convs[d] = self.encoder[i](convs[d], octree, d)
      if i < self.stage_num - 1:
        convs[d-1] = self.downsample[i](convs[d], octree, d)

    curr_depth = d
    out = convs[curr_depth]
    for i in range(self.stage_num):
      d = curr_depth + i
      out = self.decoder[i](out, octree, d)
      if i < self.stage_num - 1:
        out = self.upsample[i](out, octree, d)
        out = out + convs[d+1]  # skip connections

    return out


class Encoding(torch.nn.Module):

  def __init__(self, channels: List[int], n_node_type: int = -1, **kwargs):
    super().__init__()
    self.channels = channels
    self.n_node_type = n_node_type

    self.stage_num = len(self.channels)
    self.n_edge_type = 7
    self.norm_type = 'group_norm'
    self.act_type = 'gelu'
    self.use_checkpoint = True

    channels = [self.channels[0]] + self.channels
    self.downsample = torch.nn.ModuleList([nn.GraphDownsample(
        channels[i], channels[i+1], self.norm_type, self.act_type)
        for i in range(self.stage_num)])
    n_node_types = [self.n_node_type - i - 1 for i in range(self.stage_num)]
    self.conv = torch.nn.ModuleList([nn.GraphConvNormAct(
        self.channels[i], self.channels[i], self.n_edge_type,
        n_node_types[i], self.norm_type, self.act_type)
        for i in range(self.stage_num)])

  def checkpoint_wrapper(self, module, *args):
    if self.use_checkpoint and self.training:
      return checkpoint(module, *args)
    else:
      return module(*args)

  def forward(self, data: torch.Tensor, octree: OctreeD, depth: int):
    for i in range(self.stage_num):
      d = depth - i
      data = self.checkpoint_wrapper(self.downsample[i], data, octree, d)
      data = self.checkpoint_wrapper(self.conv[i], data, octree, d - 1)
    return data


class Decoding(torch.nn.Module):

  def __init__(self, channels: List[int], n_node_type: int = -1,
               out_channels: int = 4, predict_octree: bool = False, **kwargs):
    super().__init__()
    self.channels = channels
    self.n_node_type = n_node_type

    self.stage_num = len(self.channels)
    self.n_edge_type = 7
    self.norm_type = 'group_norm'
    self.act_type = 'gelu'
    self.use_checkpoint = True
    self.predict_octree = predict_octree
    self.out_channels = out_channels

    n_node_types = [n_node_type - i for i in range(self.stage_num-1, -1, -1)]
    self.conv = torch.nn.ModuleList([nn.GraphConvNormAct(
        self.channels[i], self.channels[i], self.n_edge_type,
        n_node_types[i], self.norm_type, self.act_type)
        for i in range(self.stage_num)])
    channels = [self.channels[0]] + self.channels
    self.upsample = torch.nn.ModuleList([nn.GraphUpsample(
        channels[i], channels[i+1], self.norm_type, self.act_type)
        for i in range(self.stage_num)])
    self.graph_pad = nn.GraphPad()

    mid_channels = 32
    self.regress = torch.nn.ModuleList([nn.Prediction(
        channels[i], mid_channels, self.out_channels, self.norm_type,
        self.act_type) for i in range(self.stage_num + 1)])
    if self.predict_octree:
      self.predict = torch.nn.ModuleList([nn.Prediction(
          channels[i], mid_channels, 2, self.norm_type, self.act_type)
          for i in range(self.stage_num + 1)])

  def forward(self, data: torch.Tensor, octree: OctreeD, depth: int,
              update_octree: bool = False):
    logits, signals = dict(), dict()
    for i in range(self.stage_num + 1):
      d = depth + i
      if i > 0:
        data = checkpoint(self.upsample[i-1], data, octree, d-1)
        data = checkpoint(self.conv[i-1], data, octree, d)

      # predict the splitting label and signal
      if self.predict_octree:
        logit = checkpoint(self.predict[i], data, octree, d)
        nnum = octree.nnum[d]
        logits[d] = logit[-nnum:]

      # regress signals and pad zeros to non-leaf nodes
      signal = checkpoint(self.regress[i], data, octree, d)
      signals[d] = self.graph_pad(signal, octree, d)

      # update the octree according to predicted labels
      if update_octree and self.predict_octree:
        split = logits[d].argmax(1).int()
        octree.octree_split(split, d)
        if i < self.stage_num - 1:
          octree.octree_grow(d + 1)

    return {'logits': logits, 'signals': signals, 'octree_out': octree}


class DiagonalGaussianDistribution(object):

  def __init__(self, parameters, deterministic=False):
    super().__init__()
    self.parameters = parameters
    self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
    self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
    self.deterministic = deterministic
    self.std = torch.exp(0.5 * self.logvar)
    self.var = torch.exp(self.logvar)
    if self.deterministic:
      self.var = torch.zeros_like(self.mean).to(device=self.parameters.device)
      self.std = self.var.clone()

  def sample(self):
    x = (self.mean +
         self.std * torch.randn(self.mean.shape, device=self.parameters.device))
    return x

  def kl(self, other=None):
    if self.deterministic:
      out = torch.Tensor([0.])
    else:
      if other is None:
        out = 0.5 * (torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar)
      else:
        out = 0.5 * torch.sum(
            torch.pow(self.mean - other.mean, 2) / other.var +
            self.var / other.var - 1.0 - self.logvar + other.logvar,
            dim=[1, 2, 3, 4])
    return out

  def nll(self, sample, dims=[1, 2, 3, 4]):
    if self.deterministic:
      out = torch.Tensor([0.])
    else:
      logtwopi = 1.8378770664093453  # np.log(2.0 * np.pi)
      out = 0.5 * torch.sum(
          logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
          dim=dims)
    return out

  def mode(self):
    return self.mean


class Encoder(torch.nn.Module):

  def __init__(self, in_channels: int, n_node_type: int = 7,
               enc_channels: List[int] = [32, 64],
               net_channels: List[int] = [64, 128, 256],
               resblk_nums: List[int] = [2, 2, 2], **kwargs):
    super().__init__()
    self.in_channels = in_channels
    self.n_node_type = n_node_type
    self.enc_channels = enc_channels
    self.net_chennels = net_channels
    self.resblk_nums = resblk_nums
    self.delta_depth = len(self.enc_channels)
    self.n_edge_type = 7

    self.conv1 = nn.GraphConvNormAct(
        self.in_channels, self.enc_channels[0], self.n_edge_type,
        self.n_node_type, norm_type='group_norm', act_type='gelu')
    self.encoding = Encoding(self.enc_channels, self.n_node_type)
    self.net = TinyNet(
        self.net_chennels, self.resblk_nums, resblk_type='basic', bottleneck=2,
        n_node_type=self.n_node_type - self.delta_depth)

  def forward(self, data: torch.Tensor, octree: OctreeD, depth: int):
    data = self.conv1(data, octree, depth)
    data = self.encoding(data, octree, depth)
    data = self.net(data, octree, depth - self.delta_depth)
    return data


class Decoder(torch.nn.Module):

  def __init__(self, out_channels: int, n_node_type: int = 7,
               dec_channels: List[int] = [64, 32],
               net_channels: List[int] = [64, 128, 256],
               resblk_nums: List[int] = [2, 2, 2],
               predict_octree: bool = False, **kwargs):
    super().__init__()
    self.out_channels = out_channels
    self.n_node_type = n_node_type
    self.dec_channels = dec_channels
    self.net_chennels = net_channels
    self.resblk_nums = resblk_nums
    self.predict_octree = predict_octree

    self.net = TinyNet(
        self.net_chennels, self.resblk_nums, resblk_type='basic', bottleneck=2,
        n_node_type=self.n_node_type-len(self.dec_channels))
    self.decoding = Decoding(
        self.dec_channels, self.n_node_type, self.out_channels,
        self.predict_octree)

  def forward(self, data: torch.Tensor, octree: OctreeD, depth: int,
              update_octree: bool = False):
    data = self.net(data, octree, depth)
    output = self.decoding(data, octree, depth, update_octree)
    return output


class GraphVAE(torch.nn.Module):

  def __init__(self, in_channels: int, n_node_type: int = 7,
               code_channels: int = 3, out_channels: int = 4,
               feature: str = 'ND', **kwargs):
    super().__init__()

    self.feature = feature
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.n_node_type = n_node_type
    self.config_network()

    self.encoder = Encoder(
        self.in_channels, self.n_node_type, self.enc_channels,
        self.enc_net_channels, self.enc_resblk_nums)
    self.decoder = Decoder(
        self.out_channels, self.n_node_type, self.dec_channels,
        self.dec_net_channels, self.dec_resblk_nums, predict_octree=True)
    self.neural_mpu = mpu.NeuralMPU()

    self.code_channels = code_channels
    self.pre_kl_conv = nn.Conv1x1(
        self.enc_channels[-1], 2 * self.code_channels, use_bias=True)
    self.post_kl_conv = nn.Conv1x1(
        self.code_channels, self.dec_channels[0], use_bias=True)

  def config_network(self):
    self.enc_channels = [32, 64]
    self.enc_net_channels = [64, 128, 256]
    self.enc_resblk_nums = [1, 1, 1]

    self.dec_channels = [64, 32]
    self.dec_net_channels = [64, 128, 256]
    self.dec_resblk_nums = [2, 2, 2]

  def forward(self, octree_in: OctreeD, octree_out: OctreeD,
              pos: torch.Tensor = None, update_octree: bool = False):
    depth = octree_in.depth
    data = octree_in.get_input_feature(feature=self.feature)
    conv = self.encoder(data, octree_in, depth)

    code = self.pre_kl_conv(conv)
    posterior = DiagonalGaussianDistribution(code)
    z = posterior.sample()
    data = self.post_kl_conv(z)

    curr_depth = depth - self.encoder.delta_depth
    output = self.decoder(data, octree_out, curr_depth, update_octree)

    output['kl_loss'] = posterior.kl().mean()
    output['code_max'] = z.max()
    output['code_min'] = z.min()

    # setup mpu
    depth_out = octree_out.depth
    self.neural_mpu.setup(output['signals'], octree_out, depth_out)

    # compute function value with mpu
    if pos is not None:
      output['mpus'] = self.neural_mpu(pos)

    # create the mpu wrapper
    output['neural_mpu'] = lambda pos: self.neural_mpu(pos)[depth_out]
    return output
