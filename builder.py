# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------


import datasets
import ognn


def get_model(flags):
  # params = [flags.depth, flags.channel, flags.nout,
  #           flags.full_depth, flags.depth_out]
  # if flags.name == 'graph_ounet' or \
  #    flags.name == 'graph_unet' or \
  #    flags.name == 'graph_ae':
  #   params.append(flags.resblock_type)
  #   params.append(flags.bottleneck)

  # if flags.name == 'octree_ounet':
  #   model = models.octree_ounet.OctreeOUNet(*params)
  if flags.name == 'graph_ounet':
    model = ognn.GraphOUNet(flags.channel, flags.resblock_type)
  # elif flags.name == 'graph_unet':
  #   model = models.graph_unet.GraphUNet(*params)
  # elif flags.name == 'graph_ae':
  #   model = models.graph_ae.GraphAE(*params)
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


# def get_classification_model(flags):
#   if flags.name.lower() == 'lenet':
#     model = ocnn.LeNet(flags.depth, flags.channel, flags.nout)
#   elif flags.name.lower() == 'resnet':
#     model = ocnn.ResNet(flags.depth, flags.channel, flags.nout,
#                         flags.resblock_num)
#   elif flags.name.lower() == 'graphlenet':
#     model = models.graph_lenet.GraphLeNet(
#         flags.depth, flags.channel, flags.nout)
#   elif flags.name.lower() == 'dualgraphlenet':
#     model = models.graph_lenet.DualGraphLeNet(
#         flags.depth, flags.channel, flags.nout)
#   elif flags.name.lower() == 'graphresnet':
#     model = models.graph_resnet.GraphResNet(
#         flags.depth, flags.channel, flags.nout, flags.resblock_num)
#   else:
#     raise ValueError
#   return model


def get_loss_function(flags):
  if flags.name.lower() == 'shapenet':
    return ognn.loss.shapenet_loss
  elif flags.name.lower() == 'dfaust':
    return ognn.loss.dfaust_loss
  elif flags.name.lower() == 'synthetic_room':
    return ognn.loss.synthetic_room_loss
  else:
    raise ValueError
