# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch


def collate_func(batch):
  output = ocnn.dataset.CollateBatch(merge_points=False)(batch)

  if 'pos' in output:
    batch_idx = torch.cat([torch.ones(pos.size(0), 1) * i
                           for i, pos in enumerate(output['pos'])], dim=0)
    pos = torch.cat(output['pos'], dim=0)
    output['pos'] = torch.cat([pos, batch_idx], dim=1)

  for key in ['grad', 'sdf', 'occu', 'weight']:
    if key in output:
      output[key] = torch.cat(output[key], dim=0)

  return output
