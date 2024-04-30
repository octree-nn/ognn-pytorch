# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------


import datasets
import ognn


class GraphOUNetR(ognn.GraphOUNet):

  def config_network(self):
    self.n_node_type = 5
    self.encoder_blk_nums = [3, 3, 3, 3, 3]
    self.decoder_blk_nums = [3, 3, 3, 3, 3]
    self.encoder_channels = [32, 64, 128, 256, 256]
    self.decoder_channels = [256, 256, 128, 64, 32]


def get_model(flags):
  if flags.name == 'graph_ounet':
    model = ognn.GraphOUNet(**flags)
  elif flags.name == 'graph_ounetr':
    model = GraphOUNetR(**flags)
  elif flags.name == 'graph_unet':
    model = ognn.GraphUNet(**flags)
  elif flags.name == 'graph_ae':
    model = ognn.GraphAE(**flags)
  elif flags.name == 'graph_vae':
    model = ognn.GraphVAE(**flags)
  else:
    raise ValueError
  return model


def get_dataset(flags):
  if flags.name.lower() == 'shapenet':
    return datasets.get_shapenet_dataset(flags)
  elif flags.name.lower() == 'shapenet_vae':
    return datasets.get_shapenet_vae_dataset(flags)
  elif flags.name.lower() == 'pointcloud':
    return datasets.get_pointcloud_dataset(flags)
  elif flags.name.lower() == 'singlepointcloud':
    return datasets.get_singlepointcloud_dataset(flags)
  elif flags.name.lower() == 'pointcloud_eval':
    return datasets.get_pointcloud_eval_dataset(flags)
  elif flags.name.lower() == 'synthetic_room':
    return datasets.get_synthetic_room_dataset(flags)
  else:
    raise ValueError


def get_loss_function(flags):
  if flags.name.lower() == 'shapenet':
    return ognn.loss.shapenet_loss
  elif flags.name.lower() == 'dfaust':
    return ognn.loss.dfaust_loss
  elif flags.name.lower() == 'synthetic_room':
    return ognn.loss.synthetic_room_loss
  elif flags.name.lower() == 'shapenet_vae':
    return ognn.loss.shapenet_vae_loss
  elif flags.name.lower() == 'shapenet_vae_color_loss':
    return ognn.loss.shapenet_vae_color_loss
  else:
    raise ValueError
