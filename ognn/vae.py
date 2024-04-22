# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from typing import List

from ognn import nn
from ognn import mpu
from ognn.octreed import OctreeD


class TinyEncoder(torch.nn.Module):

  def __init__(self, channels: List[int], resblk_nums: List[int],
               resblk_type: str = 'basic', bottleneck: int = 2,
               n_node_type: int = -1, **kwargs):
    super().__init__()
    n_edge_type = 7
    act_type = 'gelu'
    norm_type = 'group_norm'
    use_checkpoint = True
    self.stage_num = len(channels)
    self.delta_depth = len(channels) - 1

    n_node_types = [n_node_type - i for i in range(self.stage_num)]
    self.encoder = torch.nn.ModuleList([nn.GraphResBlocks(
        channels[i], channels[i], n_edge_type, n_node_types[i], norm_type,
        act_type, bottleneck, resblk_nums[i], resblk_type, use_checkpoint)
        for i in range(self.stage_num)])
    self.downsample = torch.nn.ModuleList([nn.GraphDownsample(
        channels[i], channels[i+1], norm_type, act_type)
        for i in range(self.stage_num - 1)])  # Note: self.stage_num - 1

  def forward(self, data: torch.Tensor, octree: OctreeD, depth: int):
    out = dict()
    out[depth] = data
    for i in range(self.stage_num):
      d = depth - i
      out[d] = self.encoder[i](out[d], octree, d)
      if i < self.stage_num - 1:
        out[d-1] = self.downsample[i](out[d], octree, d)
    return out


class TinyDecoder(torch.nn.Module):

  def __init__(self, channels: List[int], resblk_nums: List[int],
               resblk_type: str = 'basic', bottleneck: int = 2,
               n_node_type: int = -1, **kwargs):
    super().__init__()
    n_edge_type = 7
    act_type = 'gelu'
    norm_type = 'group_norm'
    use_checkpoint = True
    self.stage_num = len(channels)

    n_node_types = [n_node_type + i for i in range(self.stage_num)]
    self.decoder = torch.nn.ModuleList([nn.GraphResBlocks(
        channels[i], channels[i], n_edge_type, n_node_types[i], norm_type,
        act_type, bottleneck, resblk_nums[i], resblk_type, use_checkpoint)
        for i in range(self.stage_num)])
    self.upsample = torch.nn.ModuleList([nn.GraphUpsample(
        channels[i], channels[i+1], norm_type, act_type)
        for i in range(self.stage_num - 1)])

  def forward(self, datas: torch.Tensor, octree: OctreeD, depth: int):
    out = datas[depth]
    for i in range(self.stage_num):
      d = depth + i
      out = self.decoder[i](out, octree, d)
      if i < self.stage_num - 1:
        out = self.upsample[i](out, octree, d)
        out = out + datas[d+1]  # skip connections
    return out


class OctreeDecoder(torch.nn.Module):

  def __init__(self, channels: List[int], resblk_nums: List[int],
               resblk_type: str = 'basic', bottleneck: int = 2,
               n_node_type: int = -1, out_channels: int = 4,
               predict_octree: bool = False, **kwargs):
    super().__init__()
    n_edge_type = 7
    act_type = 'gelu'
    norm_type = 'group_norm'
    use_checkpoint = True
    mid_channels = 32
    self.stage_num = len(channels)
    self.predict_octree = predict_octree

    n_node_types = [n_node_type + i for i in range(self.stage_num)]
    self.decoder = torch.nn.ModuleList([nn.GraphResBlocks(
        channels[i], channels[i], n_edge_type, n_node_types[i], norm_type,
        act_type, bottleneck, resblk_nums[i], resblk_type, use_checkpoint)
        for i in range(self.stage_num)])
    self.upsample = torch.nn.ModuleList([nn.GraphUpsample(
        channels[i], channels[i+1], norm_type, act_type)
        for i in range(self.stage_num - 1)])
    self.graph_pad = nn.GraphPad()

    self.regress = torch.nn.ModuleList([nn.Prediction(
        channels[i], mid_channels, out_channels, norm_type, act_type)
        for i in range(self.stage_num)])
    if predict_octree:
      self.predict = torch.nn.ModuleList([nn.Prediction(
          channels[i], mid_channels, 2, norm_type, act_type)
          for i in range(self.stage_num)])

  def forward(self, data: torch.Tensor, octree: OctreeD, depth: int,
              update_octree: bool = False):
    logits, signals = dict(), dict()
    for i in range(self.stage_num):
      d = depth + i
      data = self.decoder[i](data, octree, d)

      # predict the splitting label and signal
      if self.predict_octree:
        logit = self.predict[i](data, octree, d)
        nnum = octree.nnum[d]
        logits[d] = logit[-nnum:]

      # regress signals and pad zeros to non-leaf nodes
      signal = self.regress[i](data, octree, d)
      signals[d] = self.graph_pad(signal, octree, d)

      # update the octree according to predicted labels
      if update_octree and self.predict_octree:
        split = logits[d].argmax(1).int()
        octree.octree_split(split, d)
        if i < self.stage_num - 1:
          octree.octree_grow(d + 1)

      # upsample
      if i < self.stage_num - 1:
        data = self.upsample[i](data, octree, d)

    return {'logits': logits, 'signals': signals, 'octree_out': octree}


class TinyNet(torch.nn.Module):

  def __init__(self, channels: List[int], resblk_nums: List[int],
               resblk_type: str = 'basic', bottleneck: int = 2,
               n_node_type: int = -1, **kwargs):
    super().__init__()
    self.encoder = TinyEncoder(
        channels, resblk_nums, resblk_type, bottleneck, n_node_type)
    self.decoder = TinyDecoder(
        channels[::-1], resblk_nums[::-1], resblk_type, bottleneck,
        n_node_type - self.encoder.delta_depth)

  def forward(self, data: torch.Tensor, octree: OctreeD, depth: int):
    encs = self.encoder(data, octree, depth)
    out = self.decoder(encs, octree, depth - self.encoder.delta_depth)
    return out


class Encoder(torch.nn.Module):

  def __init__(self, in_channels: int, n_node_type: int = 7,
               enc_channels: List[int] = [24, 32, 64],
               enc_resblk_nums: List[int] = [1, 1, 1],
               net_channels: List[int] = [64, 128, 256],
               net_resblk_nums: List[int] = [1, 2, 2], **kwargs):
    super().__init__()
    self.delta_depth = len(enc_channels) - 1

    n_edge_type = 7
    act_type = 'gelu'
    norm_type = 'group_norm'
    resblk_type = 'basic'

    self.conv1 = nn.GraphConvNormAct(
        in_channels, enc_channels[0], n_edge_type, n_node_type,
        norm_type, act_type)
    self.encoding = TinyEncoder(
        enc_channels, enc_resblk_nums, resblk_type, bottleneck=4,
        n_node_type=n_node_type)
    self.net = TinyNet(
        net_channels, net_resblk_nums, resblk_type, bottleneck=2,
        n_node_type=n_node_type - self.delta_depth)

  def forward(self, data: torch.Tensor, octree: OctreeD, depth: int):
    conv = self.conv1(data, octree, depth)
    convs = self.encoding(conv, octree, depth)
    depth = depth - self.delta_depth
    data = self.net(convs[depth], octree, depth)
    return data


class Decoder(torch.nn.Module):

  def __init__(self, out_channels: int, n_node_type: int = 7,
               dec_channels: List[int] = [64, 32, 24],
               dec_resblk_nums: List[int] = [1, 1, 1],
               net_channels: List[int] = [64, 128, 256],
               net_resblk_nums: List[int] = [1, 2, 2],
               predict_octree: bool = False, **kwargs):
    super().__init__()

    self.delta_depth = len(dec_channels) - 1
    n_node_type = n_node_type - self.delta_depth
    self.net = TinyNet(
        net_channels, net_resblk_nums, resblk_type='basic',
        bottleneck=2, n_node_type=n_node_type)
    self.decoding = OctreeDecoder(
        dec_channels, dec_resblk_nums, resblk_type='basic',
        bottleneck=4, n_node_type=n_node_type, out_channels=out_channels,
        predict_octree=predict_octree)

  def forward(self, data: torch.Tensor, octree: OctreeD, depth: int,
              update_octree: bool = False):
    data = self.net(data, octree, depth)
    output = self.decoding(data, octree, depth, update_octree)
    return output


class DiagonalGaussianDistribution(torch.nn.Module):

  def __init__(self, parameters: torch.Tensor, deterministic=False):
    super().__init__()
    self.deterministic = deterministic
    self.parameters = parameters

    self.device = parameters.device
    self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
    self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
    self.std = torch.exp(0.5 * self.logvar)
    self.var = torch.exp(self.logvar)
    if self.deterministic:
      self.var = torch.zeros_like(self.mean).to(device=self.device)
      self.std = self.var.clone()

  def sample(self):
    x = self.mean + self.std * torch.randn(self.mean.shape, device=self.device)
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


class GraphVAE(torch.nn.Module):

  def __init__(self, in_channels: int, n_node_type: int = 7,
               code_channels: int = 3, out_channels: int = 4,
               feature: str = 'ND', **kwargs):
    super().__init__()
    self.feature = feature
    self.config_network()

    self.encoder = Encoder(
        in_channels, n_node_type, self.enc_channels, self.enc_resblk_nums,
        self.enc_net_channels, self.enc_net_resblk_nums)
    self.decoder = Decoder(
        out_channels, n_node_type, self.dec_channels, self.dec_resblk_nums,
        self.dec_net_channels, self.dec_net_resblk_nums, predict_octree=True)

    self.neural_mpu = mpu.NeuralMPU()
    self.pre_kl_conv = nn.Conv1x1(
        self.enc_channels[-1], 2 * code_channels, use_bias=True)
    self.post_kl_conv = nn.Conv1x1(
        code_channels, self.dec_channels[0], use_bias=True)

  def config_network(self):
    self.enc_channels = [24, 32, 64]
    self.enc_resblk_nums = [1, 1, 1]
    self.enc_net_channels = [64, 128, 256]
    self.enc_net_resblk_nums = [1, 2, 2]

    self.dec_channels = [64, 32, 24]
    self.dec_resblk_nums = [1, 1, 1]
    self.dec_net_channels = [64, 128, 256]
    self.dec_net_resblk_nums = [1, 2, 2]

  def forward(self, octree_in: OctreeD, octree_out: OctreeD,
              pos: torch.Tensor = None, update_octree: bool = False):
    code = self.extract_code(octree_in)
    posterior = DiagonalGaussianDistribution(code)
    z = posterior.sample()
    code_depth = octree_in.depth - self.encoder.delta_depth
    # The input paramter `octree_out` is not used. It is just for compatibility
    # with the other models.
    output = self.decode_code(z, code_depth, octree_in, pos, update_octree)

    output['kl_loss'] = posterior.kl().mean()
    output['code_max'] = z.max()
    output['code_min'] = z.min()
    return output

  def extract_code(self, octree_in: OctreeD):
    depth = octree_in.depth
    data = octree_in.get_input_feature(feature=self.feature)
    conv = self.encoder(data, octree_in, depth)
    code = self.pre_kl_conv(conv)    # project features to the vae code
    return code

  def decode_code(self, code: torch.Tensor, code_depth: int, octree: OctreeD,
                  pos: torch.Tensor = None, update_octree: bool = False):
    data = self.post_kl_conv(code)   # project the vae code to features
    output = self.decoder(data, octree, code_depth, update_octree)

    # setup mpu
    depth_out = octree.depth
    self.neural_mpu.setup(output['signals'], octree, depth_out)

    # compute function value with mpu
    if pos is not None:
      output['mpus'] = self.neural_mpu(pos)

    # create the mpu wrapper
    output['neural_mpu'] = lambda pos: self.neural_mpu(pos)[depth_out]
    return output
