# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------


import datasets
import ognn


def get_model(flags):
  if flags.name == 'graph_ounet':
    model = ognn.GraphOUNet(flags.channel, flags.resblock_type)
  elif flags.name == 'graph_unet':
    model = ognn.GraphUNet(flags.channel, flags.resblock_type)
  elif flags.name == 'graph_ae':
    model = ognn.GraphAE(flags.channel, flags.resblock_type)
  else:
    raise ValueError
  return model


def get_dataset(flags):
  if flags.name.lower() == 'shapenet':
    return datasets.get_shapenet_dataset(flags)
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
  else:
    raise ValueError
